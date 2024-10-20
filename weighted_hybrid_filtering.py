import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

class HybridMatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors, movie_embeddings, embedding_dim):
        super(HybridMatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.item_bias = nn.Embedding(n_items, 1)
        self.user_bias = nn.Embedding(n_users, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        self.movie_embeddings = nn.Parameter(torch.tensor(movie_embeddings, dtype=torch.float32), requires_grad=False)
        
        self.w_cf = nn.Parameter(torch.tensor(0.5))
        self.w_cb = nn.Parameter(torch.tensor(0.5))
        
        self.content_predictor = nn.Linear(embedding_dim, 1)

    def forward(self, user_idx, item_idx):
        user_h = self.user_factors(user_idx)
        item_h = self.item_factors(item_idx)
        cf_pred = (user_h * item_h).sum(dim=1)
        
        cb_pred = self.content_predictor(self.movie_embeddings[item_idx]).squeeze()
        
        prediction = self.w_cf * cf_pred + self.w_cb * cb_pred
        
        item_bias = self.item_bias(item_idx).squeeze()
        user_bias = self.user_bias(user_idx).squeeze()
        prediction += item_bias + user_bias + self.global_bias
        
        return prediction

def train_model(model, train_data, n_epochs, batch_size, patience=10, n_folds=5,
                eta_U=0.01, eta_I=0.01, lambda_U=0.1, lambda_I=0.1):
    optimizer = torch.optim.Adam([
        {'params': model.user_factors.parameters(), 'lr': eta_U},
        {'params': model.item_factors.parameters(), 'lr': eta_I},
        {'params': list(model.item_bias.parameters()) + 
                   list(model.user_bias.parameters()) + 
                   list(model.content_predictor.parameters()) + 
                   [model.global_bias, model.w_cf, model.w_cb], 'lr': max(eta_U, eta_I)}
    ])
    mse_criterion = nn.MSELoss()
    
    gkf = GroupKFold(n_splits=n_folds)
    
    best_avg_rmse = float('inf')
    patience_counter = 0
    best_model = None
    
    for epoch in tqdm(range(n_epochs), desc="Training Epochs"):
        model.train()
        total_loss = 0
        fold_rmses = []
        
        for _, (train_idx, val_idx) in enumerate(gkf.split(train_data, groups=train_data[:, 0])):
            fold_train_data = train_data[train_idx]
            fold_val_data = train_data[val_idx]
            
            # Train on fold
            for i in range(0, len(fold_train_data), batch_size):
                batch = fold_train_data[i:i+batch_size]
                user_idx = torch.LongTensor(batch[:, 0])
                item_idx = torch.LongTensor(batch[:, 1])
                ratings = torch.FloatTensor(batch[:, 2])
                
                optimizer.zero_grad()
                predictions = model(user_idx, item_idx)
                mse_loss = mse_criterion(predictions, ratings)
                rmse_loss = torch.sqrt(mse_loss + 1e-8)  # Adding small epsilon for numerical stability
                
                l2_reg_U = lambda_U * model.user_factors.weight.norm(2)
                l2_reg_I = lambda_I * model.item_factors.weight.norm(2)
                loss = rmse_loss + l2_reg_U + l2_reg_I
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validate on fold
            fold_rmse = test_model(model, fold_val_data)
            fold_rmses.append(fold_rmse)
        
        avg_loss = total_loss / (len(train_data) / batch_size)
        avg_rmse = np.mean(fold_rmses)
        
        if avg_rmse < best_avg_rmse:
            best_avg_rmse = avg_rmse
            patience_counter = 0
            best_model = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train RMSE: {avg_loss:.4f}, Avg Val RMSE: {avg_rmse:.4f}")
            print(f"CF weight: {model.w_cf.item():.4f}, CB weight: {model.w_cb.item():.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if best_model is not None:
        model.load_state_dict(best_model)
    
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


