import torch
import torch.nn as nn

class RecommenderNet(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model.
    Learns latent representations for users and items to predict ratings.
    """
    def __init__(self, num_users, num_items, embedding_size=32):
        super(RecommenderNet, self).__init__()
        
        # User Embedding: Maps each user ID to a vector of size 'embedding_size'
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        
        # Item Embedding: Maps each product ID to a vector of size 'embedding_size'
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        
        # Multi-Layer Perceptron (MLP)
        # We concatenate user and item vectors, so input size is embedding_size * 2
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Regularization to prevent overfitting
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Final output: Predicted rating (e.g., 1.0 to 5.0)
        )

    def forward(self, user_indices, item_indices):
        """
        Forward pass logic.
        Args:
            user_indices (tensor): Batch of user IDs
            item_indices (tensor): Batch of item IDs
        Returns:
            prediction (tensor): Predicted rating
        """
        # Lookup vectors
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)
        
        # Concatenate: [User Vector, Item Vector]
        combined = torch.cat([user_vec, item_vec], dim=-1)
        
        # Pass through the MLP
        prediction = self.fc_layers(combined)
        
        # Use squeeze to return a 1D tensor of ratings
        return prediction.squeeze()