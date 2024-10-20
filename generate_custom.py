
import numpy as np
from tqdm import tqdm
import argparse
import torch
from embeddings import save_embeddings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_train.npy",
                        help="Name of the npy of the ratings table to complete")
    parser.add_argument("--model", type=str, default="gd_mf",
                        help="Name of the model")

    args = parser.parse_args()

    save_embeddings()

    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')


    n_users, n_items = table.shape
    n_factors = 20   # K

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model == "gd_mf":
        from gd_mf import MatrixFactorization
        model = MatrixFactorization(table, n_factors, mu_I=0.1, lambda_U=0.05, eta_I=0.001, xi_U=0.005, batch_size=64, n_epochs=50, device=device)
        model.train()
        with torch.no_grad():
            table = torch.matmul(model.I, model.U.T).cpu().numpy()
            
    elif args.model == "mlp_mf":
        import torch
        from mlp_mf import MLPMatrixFactorization, train_model

        movie_embeddings = np.load('movie_embeddings.npy')
        embedding_dim = movie_embeddings.shape[1]
        
        train_data = [(i, j, table[i, j]) for i, j in zip(*np.where(~np.isnan(table)))]
        train_data = np.array(train_data)
        
        model = MLPMatrixFactorization(n_users, n_items, n_factors, movie_embeddings, embedding_dim)
        train_model(model, train_data, n_epochs=70, learning_rate=0.001, batch_size=64)
        model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(n_users)):
                user_idx = torch.LongTensor([i] * n_items)
                item_idx = torch.LongTensor(range(n_items))
                
                predictions = model(user_idx, item_idx).numpy()
                table[i, :] = predictions

    elif args.model == "hybrid_mf":
        import torch
        from weighted_hybrid_filtering import HybridMatrixFactorization, train_model

        movie_embeddings = np.load('movie_embeddings.npy')
        embedding_dim = movie_embeddings.shape[1]
        
        train_data = [(i, j, table[i, j]) for i, j in zip(*np.where(~np.isnan(table)))]
        train_data = np.array(train_data)
        
        model = HybridMatrixFactorization(n_users, n_items, n_factors, movie_embeddings, embedding_dim)
        model = train_model(model, train_data, n_epochs=50, batch_size=64, patience=10, n_folds=5,
                            eta_U=0.005, eta_I=0.001, lambda_U=0.05, lambda_I=0.1)
        model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(n_users)):
                user_idx = torch.LongTensor([i] * n_items)
                item_idx = torch.LongTensor(range(n_items))
                
                predictions = model(user_idx, item_idx).numpy()
                table[i, :] = predictions
    
    elif args.model == "pytorch_gd_mf":
        import torch
        from pytorch_gd_mf import MatrixFactorization, train_model

        train_data = [(i, j, table[i, j]) for i, j in zip(*np.where(~np.isnan(table)))]
        train_data = np.array(train_data)
        
        model = MatrixFactorization(n_users, n_items, n_factors)
        train_model(model, train_data, n_epochs=50, batch_size=64, lr=0.001, lambda_U=0.05, lambda_I=0.1)
        
        model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(n_users)):
                user_idx = torch.LongTensor([i] * n_items)
                item_idx = torch.LongTensor(range(n_items))
                
                predictions = model(user_idx, item_idx).numpy()
                table[i, :] = predictions

    else:
        raise ValueError(f"Model {args.model} not found")

    

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE
