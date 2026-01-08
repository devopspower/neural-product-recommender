import torch
import torch.nn as nn
from recommender_dataset import get_processed_data
from model import RecommenderNet
import os

# --- Configuration & Hyperparameters ---
CONFIG = {
    'file_path': 'data/product-reviews.csv',
    'batch_size': 32,
    'embedding_size': 32,
    'learning_rate': 0.001,
    'epochs': 25,
    'model_save_path': 'recommender.pth'
}

def main():
    # 1. Define Device (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Starting Recommendation Engine Pipeline ---")
    print(f"Using device: {device}")

    # 2. Ground Objectively: Load and Process Data
    # Returns loaders and the counts needed to define the Embedding layers
    train_loader, val_loader, n_users, n_items = get_processed_data(
        CONFIG['file_path'], 
        batch_size=CONFIG['batch_size']
    )

    # 3. Analyze Logically: Initialize NCF Model
    model = RecommenderNet(
        num_users=n_users, 
        num_items=n_items, 
        embedding_size=CONFIG['embedding_size']
    ).to(device)

    # 4. Define Loss and Optimizer
    # MSE is used because we are predicting the specific rating value (1-5)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # 5. Validate Rigorously: Training Loop
    print(f"Beginning training for {n_users} users and {n_items} items...")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        
        for users, items, ratings in train_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(users, items)
            loss = criterion(outputs, ratings)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for users, items, ratings in val_loader:
                users, items, ratings = users.to(device), items.to(device), ratings.to(device)
                outputs = model(users, items)
                loss = criterion(outputs, ratings)
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1:02d}/{CONFIG['epochs']}] | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    # 6. Act Precisely: Save the Model
    torch.save(model.state_dict(), CONFIG['model_save_path'])
    print(f"--- Process Complete ---")
    print(f"Model saved to: {CONFIG['model_save_path']}")

if __name__ == "__main__":
    main()