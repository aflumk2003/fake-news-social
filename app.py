import streamlit as st
import pickle
import re
import os
import urllib.request

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Fake News Detection")

# ---------------------------
# SHOW UI FIRST (IMPORTANT)
# ---------------------------
st.title("📰 Fake News Detection Using Social Media Text")
st.write("AI-powered misinformation detection")

st.markdown("---")

# ---------------------------
# DOWNLOAD MODEL SAFELY
# ---------------------------
MODEL_URL = "https://raw.githubusercontent.com/aflumk2003/fake-news-social/main/model.pkl"
VECTORIZER_URL = "https://raw.githubusercontent.com/aflumk2003/fake-news-social/main/vectorizer.pkl"

def safe_download(url, filename):
    try:
        if not os.path.exists(filename):
            st.write(f"Downloading {filename}...")
            data = urllib.request.urlopen(url).read()
            with open(filename, "wb") as f:
                f.write(data)
    except Exception as e:
        st.error(f"Download failed: {e}")
        st.stop()

safe_download(MODEL_URL, "model.pkl")
safe_download(VECTORIZER_URL, "vectorizer.pkl")

# ---------------------------
# LOAD MODEL SAFELY
# ---------------------------
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

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