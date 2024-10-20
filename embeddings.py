import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

def get_bert_embeddings(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding as the sentence embedding
    return embeddings

def save_embeddings():
    genres = np.load('namesngenre.npy')

    df = pd.DataFrame(genres, columns=['name', 'genre'])
    df['year'] = df['name'].apply(lambda x: x.split('(')[-1].replace(')', '') if x.count('(') > 0 else np.nan) # Might use year information in further experiments
    df['name'] = df['name'].apply(lambda x: x.split('(')[0].strip() if x.count('(') > 0 else x)
    names_embeddings = get_bert_embeddings(df['name'].tolist())
    genres_embeddings = get_bert_embeddings(df['genre'].tolist())

    movie_embeddings = torch.cat((names_embeddings, genres_embeddings), dim=1)
    np.save('movie_embeddings.npy', movie_embeddings.numpy())