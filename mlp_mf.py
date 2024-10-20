import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

class MLPMatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors, movie_embeddings, embedding_dim):
        super(MLPMatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.item_bias = nn.Embedding(n_items, 1)
        self.user_bias = nn.Embedding(n_users, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        self.mlp = nn.Sequential(
            nn.Linear(n_factors + embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.movie_embeddings = nn.Parameter(torch.tensor(movie_embeddings, dtype=torch.float32), requires_grad=False)

    def forward(self, user_idx, item_idx):
        user_h = self.user_factors(user_idx)
        item_h = self.item_factors(item_idx)
        item_bias = self.item_bias(item_idx).squeeze()
        user_bias = self.user_bias(user_idx).squeeze()
        
        item_features = torch.cat([item_h, self.movie_embeddings[item_idx]], dim=1)
        
        prediction = self.mlp(item_features).squeeze()
        prediction += item_bias + user_bias + self.global_bias
        
        return prediction

def train_model(model, train_data, n_epochs, learning_rate, batch_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in tqdm(range(n_epochs), desc="Training Epochs"):
        model.train()
        total_loss = 0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            user_idx = torch.LongTensor(batch[:, 0])
            item_idx = torch.LongTensor(batch[:, 1])
            ratings = torch.FloatTensor(batch[:, 2])
            
            optimizer.zero_grad()
            predictions = model(user_idx, item_idx)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(train_data) / batch_size)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
