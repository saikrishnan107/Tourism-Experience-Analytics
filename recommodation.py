import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\Sai Krishnan\\OneDrive\\Desktop\\ML Program\\Tourism Analysis\\preprocessed_data.csv")
    return df

# Compute item-item similarity matrix
@st.cache_data
def prepare_item_similarity(df):
    item_matrix = df.pivot_table(index='UserId', columns='AttractionId', values='Rating').fillna(0)
    item_similarity = cosine_similarity(item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=item_matrix.columns, columns=item_matrix.columns)
    return item_matrix, item_similarity_df

# Recommendation function
def recommend_items(user_id, item_matrix, item_similarity_df, top_n=5):
    if user_id not in item_matrix.index:
        return []

    user_ratings = item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index

    scores = pd.Series(dtype=float)
    for item in rated_items:
        similar_scores = item_similarity_df[item] * user_ratings[item]
        scores = scores.add(similar_scores, fill_value=0)

    scores = scores.drop(rated_items, errors='ignore') 
    return scores.sort_values(ascending=False).head(top_n)

# Streamlit UI
st.set_page_config(page_title="Tourist Recommender", layout="centered")
st.title("🧭 Tourist Attraction Recommender System")

# Load data
df = load_data()
item_matrix, item_similarity_df = prepare_item_similarity(df)

# User selection
user_ids = df['UserId'].dropna().unique().tolist()
selected_user = st.selectbox("Select User ID", sorted(user_ids))

# Generate recommendations
if st.button("Get Recommendations"):
    recommendations = recommend_items(selected_user, item_matrix, item_similarity_df)

    if recommendations.empty:
        st.warning("No recommendations found for this user.")
    else:
        st.subheader("Top Recommended Attractions:")
        attraction_lookup = df[['AttractionId', 'Attraction']].drop_duplicates().set_index('AttractionId')

        for attraction_id, score in recommendations.items():
            name = attraction_lookup.loc[attraction_id, 'Attraction'] if attraction_id in attraction_lookup.index else "Unknown"
            st.write(f"🎯 {name} (ID: {attraction_id}) — Score: {round(score, 2)}")