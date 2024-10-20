import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

# Matrix Factorization using Gradient Descent
class MatrixFactorization:
    def __init__(self, R, K, mu_I=0.1, lambda_U=0.1, eta_I=0.001, xi_U=0.001, batch_size=64, n_epochs=1000, device='cpu'):
        self.R = torch.tensor(R, dtype=torch.float32, device=device)
        self.K = K 
        self.mu_I = mu_I  # regularization parameter for I
        self.lambda_U = lambda_U  # regularization parameter for U 
        self.eta_I = eta_I  # learning rate 
        self.xi_U = xi_U  # learning rate 
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.n_users, self.n_items = R.shape

        self.I_start = torch.rand(self.n_users, K, device=device) 
        self.U_start = torch.rand(self.n_items, K, device=device) 

        self.U = torch.tensor(self.U_start, dtype=torch.float32, requires_grad=True, device=device) 
        self.I = torch.tensor(self.I_start, dtype=torch.float32, requires_grad=True, device=device) 

        self.S = np.where(~np.isnan(R))  # observed ratings, i.e., non-missing values

        self.optimizer = torch.optim.Adam([
            {'params': self.I, 'lr': self.eta_I}, 
            {'params': self.U, 'lr': self.xi_U} 
        ])        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)

    def predict(self, i, j):
        return torch.sum(self.I[i] * self.U[j], dim=1)

    def gradient_I(self, i, j, error):
        return -2 * error[:, None] * self.U[j] + 2 * self.mu_I * self.I[i]

    def gradient_U(self, i, j, error):
        return -2 * error[:, None] * self.I[i] + 2 * self.lambda_U * self.U[j]

    def process_batch(self, batch):
        i = torch.tensor(self.S[0][batch], device=self.device)
        j = torch.tensor(self.S[1][batch], device=self.device)
        error = self.R[i, j] - self.predict(i, j)
        grad_I = torch.zeros_like(self.I)
        grad_U = torch.zeros_like(self.U)
        grad_I.index_add_(0, i, self.gradient_I(i, j, error))
        grad_U.index_add_(0, j, self.gradient_U(i, j, error))
        return grad_I, grad_U

    def train(self, n_folds=5):
        n_observed = len(self.S[0])
        indices = np.arange(n_observed)
        best_rmse = float('inf')
        patience = 10
        no_improve = 0

        gkf = GroupKFold(n_splits=n_folds)
        groups = self.S[0]  # group by users

        for epoch in tqdm(range(1, self.n_epochs + 1), desc="Training Epochs"):
            fold_rmses = []

            for train_idx, val_idx in gkf.split(indices, groups=groups):
                train_indices = indices[train_idx]
                val_indices = indices[val_idx]

                # Training
                np.random.shuffle(train_indices)
                batches = [train_indices[i:i + self.batch_size] for i in range(0, len(train_indices), self.batch_size)]

                total_grad_I = torch.zeros_like(self.I)
                total_grad_U = torch.zeros_like(self.U)
                for batch in batches:
                    grad_I, grad_U = self.process_batch(batch)
                    total_grad_I += grad_I
                    total_grad_U += grad_U

                self.optimizer.zero_grad()
                self.I.grad = total_grad_I / len(batches)
                self.U.grad = total_grad_U / len(batches)
                self.optimizer.step()

                # Validation
                val_rmse = self.rmse(val_indices)
                fold_rmses.append(val_rmse)

            avg_rmse = np.mean(fold_rmses)
            self.scheduler.step(avg_rmse)

            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Avg Val RMSE: {avg_rmse:.4f}")

    def rmse(self, indices):
        i = torch.tensor(self.S[0][indices], device=self.device)
        j = torch.tensor(self.S[1][indices], device=self.device)
        squared_errors = (self.R[i, j] - self.predict(i, j))**2
        return torch.sqrt(torch.mean(squared_errors)).item()

