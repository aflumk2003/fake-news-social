import streamlit as st
import pickle
import re

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------
# TITLE (SHOW IMMEDIATELY)
# ---------------------------
st.markdown(
    "<h1 style='text-align:center; color:#00ffd5;'>📰 Fake News Detection Using Social Media Text</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>AI-powered misinformation detection</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------
# LOAD MODEL (CACHED 🔥)
# ---------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

try:
    model, vectorizer = load_model()
except:
    st.error("❌ Model not found. Make sure model.pkl and vectorizer.pkl are in the folder.")
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
st.markdown("### ✍️ Enter Text")

user_input = st.text_area(
    "",
    placeholder="Paste tweet, WhatsApp forward, or news...",
    height=150
)

# ---------------------------
# BUTTON
# ---------------------------
if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Enter some text")
    else:
        cleaned = clean_text(user_input)

        input_vec = vectorizer.transform([cleaned])

        prediction = model.predict(input_vec)[0]
        probability = model.predict_proba(input_vec)[0]

        confidence = max(probability) * 100

        st.markdown("---")

        if confidence < 55:
            st.warning(f"🤔 Uncertain ({confidence:.2f}%)")
        elif prediction == 1:
            st.success(f"✅ Real News ({confidence:.2f}%)")
        else:
            st.error(f"🚨 Fake News ({confidence:.2f}%)")

        st.progress(int(confidence))