import streamlit as st
import pandas as pd
import torch
import pickle
from inference import get_recommendations

# --- Page Configuration ---
st.set_page_config(page_title="Amazon AI Recommender", page_icon="🛒", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("🛒 Neural Product Recommendation Engine")
st.write("Leveraging **Neural Collaborative Filtering** to discover latent user-product relationships.")

# --- Sidebar / Controls ---
st.sidebar.header("User Selection")

@st.cache_data
def get_user_list():
    df = pd.read_csv('data/product-reviews.csv')
    return sorted(df['reviews.username'].dropna().unique().tolist())

user_list = get_user_list()
selected_user = st.sidebar.selectbox("Select a User to Generate Recommendations:", user_list, index=user_list.index("Ricky") if "Ricky" in user_list else 0)
num_recs = st.sidebar.slider("Number of Recommendations", 3, 10, 5)

# --- Data Overview ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("👤 User Profile")
    st.metric("Username", selected_user)
    
    # Show user history
    df_raw = pd.read_csv('data/product-reviews.csv')
    user_history = df_raw[df_raw['reviews.username'] == selected_user][['name', 'reviews.rating']]
    
    st.write(f"**Previous Purchases ({len(user_history)}):**")
    st.table(user_history)

with col2:
    st.subheader("🎯 Personalized Top Picks")
    
    if st.button("Generate Recommendations"):
        with st.spinner('Analyzing latent features...'):
            recommendations = get_recommendations(selected_user, top_k=num_recs)
            
            if isinstance(recommendations, str):
                st.error(recommendations)
            else:
                # Convert recs to DataFrame for better display
                rec_df = pd.DataFrame(recommendations)
                
                # Fetch product names for the IDs (joining with raw data)
                product_names = df_raw[['id', 'name']].drop_duplicates('id')
                rec_df = rec_df.merge(product_names, left_on='product_id', right_on='id', how='left')
                
                # Display Results
                for idx, row in rec_df.iterrows():
                    with st.container():
                        c1, c2 = st.columns([4, 1])
                        c1.markdown(f"**{idx+1}. {row['name']}**")
                        c1.caption(f"Product ID: {row['product_id']}")
                        c2.metric("Score", f"{row['predicted_rating']}/5")
                        st.divider()
    else:
        st.info("Click the button above to run the Neural Network inference.")

# --- Logic Breakdown ---
st.sidebar.divider()
st.sidebar.subheader("Model Insights")
st.sidebar.write("""
- **Algorithm:** Neural Collaborative Filtering (NCF)
- **Architecture:** 32-dim User/Item Embeddings + 3-Layer MLP
- **Training Loss:** 0.87 (MSE)
""")