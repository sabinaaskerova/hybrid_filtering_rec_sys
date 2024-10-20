import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors):
        super(MatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_idx, item_idx):
        user_factors = self.user_factors(user_idx)
        item_factors = self.item_factors(item_idx)
        user_bias = self.user_bias(user_idx).squeeze()
        item_bias = self.item_bias(item_idx).squeeze()
        prediction = (user_factors * item_factors).sum(dim=1) + user_bias + item_bias + self.global_bias
        return prediction

def train_model(model, train_data, n_epochs, batch_size, lr=0.01, lambda_U=0.1, lambda_I=0.1, n_folds=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = nn.MSELoss()

    gkf = GroupKFold(n_splits=n_folds)
    groups = train_data[:, 0] 

    best_rmse = float('inf')
    patience = 10
    no_improve = 0

    for epoch in tqdm(range(n_epochs), desc="Training Epochs"):
        model.train()
        fold_rmses = []

        for train_idx, val_idx in gkf.split(train_data, groups=groups):
            train_subset = train_data[train_idx]
            val_subset = train_data[val_idx]

            total_loss = 0

            for i in range(0, len(train_subset), batch_size):
                batch = train_subset[i:i+batch_size]
                user_idx = torch.LongTensor(batch[:, 0])
                item_idx = torch.LongTensor(batch[:, 1])
                ratings = torch.FloatTensor(batch[:, 2])

                optimizer.zero_grad()
                predictions = model(user_idx, item_idx)
                mse_loss = criterion(predictions, ratings)
                l2_reg_U = lambda_U * model.user_factors.weight.norm(2)
                l2_reg_I = lambda_I * model.item_factors.weight.norm(2)
                loss = mse_loss + l2_reg_U + l2_reg_I

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (len(train_subset) / batch_size)
            val_rmse = test_model(model, val_subset)
            fold_rmses.append(val_rmse)

        avg_rmse = np.mean(fold_rmses)
        scheduler.step(avg_rmse)

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Avg Val RMSE: {avg_rmse:.4f}")

    return model

def test_model(model, test_data):
    model.eval()
    with torch.no_grad():
        user_idx = torch.LongTensor(test_data[:, 0])
        item_idx = torch.LongTensor(test_data[:, 1])
        true_ratings = torch.FloatTensor(test_data[:, 2])

        predictions = model(user_idx, item_idx)
        mse = nn.MSELoss()(predictions, true_ratings)
        rmse = torch.sqrt(mse)

    return rmse.item()
