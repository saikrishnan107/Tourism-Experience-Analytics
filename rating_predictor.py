import streamlit as st
import pandas as pd
import joblib

# Load dataset
df_cleaned = pd.read_csv('C:\\Users\\Sai Krishnan\\OneDrive\\Desktop\\ML Program\\Tourism Analysis\\preprocessed_data.csv')

# Load trained pipeline model
model = joblib.load('C:\\Users\\Sai Krishnan\\OneDrive\\Desktop\\ML Program\\Tourism Analysis\\pipeline_xgb.pkl')

st.set_page_config(page_title="Tourism Rating Predictor", layout="centered")
st.title("🌍 Tourism Rating Predictor")

# Sidebar - user inputs
st.sidebar.header("🔎 Your Preferences")

# Input widgets based on available data
continent = st.sidebar.selectbox("Continent", df_cleaned["Continent"].dropna().unique())
region = st.sidebar.selectbox("Region", df_cleaned[df_cleaned["Continent"] == continent]["Region"].dropna().unique())
country = st.sidebar.selectbox("Country", df_cleaned[(df_cleaned["Region"] == region)]["Country"].dropna().unique())
city = st.sidebar.selectbox("City", df_cleaned[(df_cleaned["Country"] == country)]["CityName"].dropna().unique())

visit_year = st.sidebar.selectbox("Visit Year", sorted(df_cleaned["VisitYear"].dropna().unique()))
visit_month = st.sidebar.selectbox("Visit Month", sorted(df_cleaned["VisitMonth"].dropna().unique()))
visit_mode = st.sidebar.selectbox("Mode of Visit", df_cleaned["VisitMode"].dropna().unique())

attraction_type = st.sidebar.selectbox("Attraction Type", df_cleaned["AttractionTypeId"].dropna().unique())
attraction_avg_rating = st.sidebar.slider("Attraction Avg Rating", min_value=1.0, max_value=5.0, step=0.1)
user_avg_rating = st.sidebar.slider("Your Avg Rating", min_value=1.0, max_value=5.0, step=0.1)

# Optional or defaulted values
user_total_visit = st.sidebar.number_input("Your Total Visits", value=5)
user_visit_count = st.sidebar.number_input("Your Visit Count", value=5)
attraction_total_visit = st.sidebar.number_input("Attraction Total Visits", value=0)

# Create input data in correct format
input_data = pd.DataFrame([{
    "Continent": continent,
    "Region": region,
    "Country": country,
    "CityName": city,
    "VisitYear": visit_year,
    "VisitMonth": visit_month,
    "VisitMode": visit_mode,
    "AttractionTypeId": attraction_type,
    "AttractionAvgRating": attraction_avg_rating,
    "UserTotalVisit": user_total_visit,
    "UserVisitCount": user_visit_count,
    "Attraction": "Unknown",  # This fixes the error
    "AttractionTotalVisit": attraction_total_visit,
    "UserAvgRating": user_avg_rating
}])

# Display input
st.subheader("📝 Your Inputs")
st.write(input_data)

# Predict rating
if st.button("🔮 Predict Rating"):
    try:
        rating = model.predict(input_data)[0]
        st.success(f"🌟 Predicted Rating: {rating:.2f} / 5")
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
