import torch
import pickle
import pandas as pd
import numpy as np
from model import RecommenderNet

def get_recommendations(user_name, top_k=5, model_path='recommender.pth', encoders_path='encoders.pkl'):
    """
    Generates personalized product recommendations for a specific user.
    """
    # 1. Load Encoders and trained Model
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    
    user_enc = encoders['user_enc']
    item_enc = encoders['item_enc']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model architecture and load weights
    model = RecommenderNet(len(user_enc.classes_), len(item_enc.classes_)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Convert Username to index
    try:
        user_idx = user_enc.transform([user_name])[0]
    except ValueError:
        return f"User '{user_name}' not found in the dataset."

    # 3. Explore Systematically: Score all products for this user
    # We create a list of all item indices to pass through the model
    all_item_indices = torch.arange(len(item_enc.classes_)).to(device)
    user_indices = torch.full_like(all_item_indices, user_idx).to(device)
    
    with torch.no_grad():
        # The model predicts a rating for every single product in the catalog
        predicted_ratings = model(user_indices, all_item_indices)
    
    # 4. Rank and Filter
    # Get the indices of the items with the highest predicted ratings
    top_indices = torch.argsort(predicted_ratings, descending=True)[:top_k]
    
    # Convert indices back to original Product IDs (or names)
    recommended_item_ids = item_enc.inverse_transform(top_indices.cpu().numpy())
    recommended_scores = predicted_ratings[top_indices].cpu().numpy()

    # 5. Format results
    results = []
    for iid, score in zip(recommended_item_ids, recommended_scores):
        results.append({"product_id": iid, "predicted_rating": round(float(score), 2)})
        
    return results

if __name__ == "__main__":
    # Test with a user from the Amazon dataset
    # Change 'Ricky' to any username present in your 'product-reviews.csv'
    target_user = "Ricky"
    recommendations = get_recommendations(target_user)
    
    if isinstance(recommendations, str):
        print(recommendations)
    else:
        print(f"--- Top 5 Recommendations for {target_user} ---")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. Product: {rec['product_id']} | Score: {rec['predicted_rating']}")