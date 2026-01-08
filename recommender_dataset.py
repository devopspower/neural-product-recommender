import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pickle

class AmazonDataset(Dataset):
    """
    Custom Dataset to hold User, Item, and Rating triplets.
    """
    def __init__(self, users, items, ratings):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

def get_processed_data(file_path, batch_size=32):
    """
    Preprocesses the Amazon Product Reviews dataset for Collaborative Filtering.
    """
    # 1. Load Data
    df = pd.read_csv(file_path)
    
    # 2. Precision: Filter critical columns and drop nulls
    # We use 'reviews.username' for Users and 'id' for Items
    df = df[['reviews.username', 'id', 'reviews.rating']].dropna()
    
    # 3. Label Encoding: Map sparse IDs to contiguous integers [0, N-1]
    # This is required for the Embedding layers to work correctly.
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    
    df['user_idx'] = user_enc.fit_transform(df['reviews.username'])
    df['item_idx'] = item_enc.fit_transform(df['id'])
    
    num_users = len(user_enc.classes_)
    num_items = len(item_enc.classes_)
    
    # 4. Save Encoders: Crucial for reversing indices back to names during inference
    with open('encoders.pkl', 'wb') as f:
        pickle.dump({'user_enc': user_enc, 'item_enc': item_enc}, f)
        
    # 5. Split Rigorously: 80% Training, 20% Validation
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    # 6. Create DataLoaders
    train_ds = AmazonDataset(
        train_df['user_idx'].values, 
        train_df['item_idx'].values, 
        train_df['reviews.rating'].values
    )
    val_ds = AmazonDataset(
        val_df['user_idx'].values, 
        val_df['item_idx'].values, 
        val_df['reviews.rating'].values
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, num_users, num_items