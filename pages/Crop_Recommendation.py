import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --------------------------
# Load Dataset & Train Model
# --------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Crop_recommendation.csv")

@st.cache_resource
def train_model(data):
    X = data.drop("label", axis=1)
    y = data["label"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

data = load_data()
model = train_model(data)

# --------------------------
# UI
# --------------------------
st.title("🌱 AI-Based Crop Recommendation System")

N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=90.0)
P = st.number_input("Phosphorous (P)", min_value=0.0, max_value=200.0, value=50.0)
K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0)

if st.button("Recommend Crop"):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)[0]
    st.success(f"✅ Recommended Crop: **{prediction}**")
