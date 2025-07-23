import streamlit as st
import pandas as pd
import joblib

# Load dataset
df_cleaned = pd.read_csv('C:\\Users\\Sai Krishnan\\OneDrive\\Desktop\\ML Program\\Tourism Analysis\\preprocessed_data.csv')

# Load trained pipeline model
model = joblib.load('C:\\Users\\Sai Krishnan\\OneDrive\\Desktop\\ML Program\\Tourism Analysis\\pipeline_xgb.pkl')

st.title("🎯 Tourism Rating Predictor")

# Sidebar for input features
st.sidebar.header("🧭 Select Your Inputs")

# --- Dependent Dropdowns ---
# 1. Continent
continent = st.sidebar.selectbox("Select Continent", df_cleaned["Continent"].dropna().unique())

# 2. Region based on continent
region_options = df_cleaned[df_cleaned["Continent"] == continent]["Region"].dropna().unique()
region = st.sidebar.selectbox("Select Region", region_options)

# 3. Country based on region
country_options = df_cleaned[
    (df_cleaned["Continent"] == continent) & 
    (df_cleaned["Region"] == region)
]["Country"].dropna().unique()
country = st.sidebar.selectbox("Select Country", country_options)

# 4. City based on country
city_options = df_cleaned[
    (df_cleaned["Continent"] == continent) & 
    (df_cleaned["Region"] == region) & 
    (df_cleaned["Country"] == country)
]["CityName"].dropna().unique()
city = st.sidebar.selectbox("Select City", city_options)

# 5. Visit details
visit_year = st.sidebar.selectbox("Select Visit Year", sorted(df_cleaned["VisitYear"].dropna().unique()))
visit_month = st.sidebar.selectbox("Select Visit Month", sorted(df_cleaned["VisitMonth"].dropna().unique()))
mode_of_visit = st.sidebar.selectbox("Select Visit Mode", df_cleaned["VisitMode"].dropna().unique())

# 6. Attraction type
attraction_type_id = st.sidebar.selectbox("Select Attraction Type", df_cleaned["AttractionTypeId"].dropna().unique())

# 7. Average rating input
avg_rating = st.sidebar.slider("Previous Average Rating", min_value=1.0, max_value=5.0, step=0.1)

# --- Prepare input data ---
input_data = pd.DataFrame({
    "Continent": [continent],
    "Region": [region],
    "Country": [country],
    "CityName": [city],
    "VisitYear": [visit_year],
    "VisitMonth": [visit_month],
    "VisitMode": [mode_of_visit],
    "AttractionTypeId": [attraction_type_id],
    "AttractionAvgRating": [avg_rating],
    "UserTotalVisit": [5],             # Dummy or default
    "UserVisitCount": [0],             # Dummy or default
    "Attraction": ["Unknown"],         # Placeholder
    "AttractionTotalVisit": [0],       # Dummy or default
    "UserAvgRating": [avg_rating]
})

# --- Show input summary ---
st.subheader("📝 Your Input Summary:")
st.write(input_data)

# --- Prediction logic ---
expected_cols = [
    'Continent', 'Region', 'Country', 'CityName', 'VisitYear', 'VisitMonth',
    'VisitMode', 'AttractionTypeId', 'AttractionAvgRating', 'UserTotalVisit',
    'UserVisitCount', 'Attraction', 'AttractionTotalVisit', 'UserAvgRating'
]

if st.button("🔮 Predict Rating"):
    missing_cols = [col for col in expected_cols if col not in input_data.columns]

    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
    else:
        try:
            # Ensure correct column order
            input_data = input_data[expected_cols]
            prediction = model.predict(input_data)
            st.success(f"🌟 Predicted Rating: **{prediction[0]:.2f} / 5**")
        except Exception as e:
            st.error(f"🚨 Error during prediction: {e}")
