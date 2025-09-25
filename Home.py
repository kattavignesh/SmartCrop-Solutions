import streamlit as st

st.set_page_config(page_title="AI Farming Assistant", layout="centered")

st.title("🌾 AI Farming Assistant")
st.write("Choose one of the features below:")

# Three buttons for navigation
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🤖 AI Farming Chatbot"):
        st.switch_page("pages/Chatbot.py")

with col2:
    if st.button("🌱 Crop Recommendation System"):
        st.switch_page("pages/Crop_Recommendation.py")

with col3:
    if st.button("🩺 Disease Predictor"):
        st.switch_page("pages/Disease_Predictor.py")
