import streamlit as st
from gpt4all import GPT4All
from dotenv import load_dotenv
from streamlit_js_eval import get_geolocation
import os
import requests

# Load API keys
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

st.title("🤖 AI Farming Chatbot")

# Load local LLaMA model
model_path = r"C:/Users/KATTA VIGNESH/AppData/Local/nomic.ai/GPT4All/Llama-3.2-3B-Instruct-Q4_0.gguf"
st.write("⏳ Loading model... please wait 1–3 minutes.")
model = GPT4All(model_path, allow_download=False)
st.success("✅ Model loaded!")

# 🌍 Get user location
st.subheader("📍 Location")
loc = get_geolocation()
if loc:
    latitude = loc["coords"]["latitude"]
    longitude = loc["coords"]["longitude"]
    st.success(f"✅ Location detected: Lat {latitude}, Lon {longitude}")
else:
    st.info("Please allow location access in your browser.")

# Weather function using lat/lon
def get_weather_by_location(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return f"🌦 Weather: {data['main']['temp']}°C, {data['weather'][0]['description']}"
    else:
        return f"⚠️ Error {response.status_code}: {response.text}"

if loc and st.button("🌤 Get My Weather"):
    weather = get_weather_by_location(latitude, longitude)
    st.success(weather)


# 💬 Chat input
user_input = st.text_input("💬 Ask me anything about farming:")

if st.button("Ask Question"):
    if user_input.strip():
        with model.chat_session():
            context = user_input
            if loc:
                context += f"\n(My location: lat={latitude}, lon={longitude})"
                weather_info = get_weather_by_location(latitude, longitude)
                context += f"\n(Current weather: {weather_info})"

            # Show loading spinner while model generates response
            with st.spinner("🤖 Loading response... please wait..."):
                answer = model.generate(context, max_tokens=150, temp=0.7)

        # Display AI response
        st.success("🤖 " + answer)
    else:
        st.warning("⚠️ Please enter a question.")
