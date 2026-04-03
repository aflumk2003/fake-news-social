import streamlit as st
import pickle
import re
import os
import urllib.request

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Fake News Detection")

st.title("📰 Fake News Detection Using Social Media Text")
st.write("AI-powered misinformation detection")

st.markdown("---")

# ---------------------------
# DOWNLOAD ONLY ONCE
# ---------------------------
MODEL_URL = "https://raw.githubusercontent.com/aflumk2003/fake-news-social/main/model.pkl"
VECTORIZER_URL = "https://raw.githubusercontent.com/aflumk2003/fake-news-social/main/vectorizer.pkl"

def download_once(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename}..."):
            data = urllib.request.urlopen(url).read()
            with open(filename, "wb") as f:
                f.write(data)

download_once(MODEL_URL, "model.pkl")
download_once(VECTORIZER_URL, "vectorizer.pkl")

# ---------------------------
# CACHE MODEL (CRITICAL 🔥)
# ---------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ---------------------------
# CLEAN TEXT
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# ---------------------------
# INPUT
# ---------------------------
user_input = st.text_area("Enter text here")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Enter some text")
    else:
        cleaned = clean_text(user_input)

        input_vec = vectorizer.transform([cleaned])

        prediction = model.predict(input_vec)[0]
        probability = model.predict_proba(input_vec)[0]

        confidence = max(probability) * 100

        if confidence < 55:
            st.warning(f"🤔 Uncertain ({confidence:.2f}%)")
        elif prediction == 1:
            st.success(f"✅ Real News ({confidence:.2f}%)")
        else:
            st.error(f"🚨 Fake News ({confidence:.2f}%)")

        st.progress(int(confidence))