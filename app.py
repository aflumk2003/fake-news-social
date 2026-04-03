import streamlit as st
import pickle
import re

# ---------------------------
# LOAD MODEL (LOCAL FILES)
# ---------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------------------
# TEXT CLEANING
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------
# UI DESIGN
# ---------------------------
st.markdown(
    "<h1 style='text-align:center; color:#00ffd5;'>📰 Fake News Detection Using Social Media Text</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:gray;'>Detect misinformation using Machine Learning & NLP</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------
# INPUT
# ---------------------------
st.markdown("### ✍️ Enter Social Media Text")

user_input = st.text_area(
    "",
    placeholder="Paste WhatsApp forward, tweet, or news headline here...",
    height=150
)

col1, col2 = st.columns(2)

with col1:
    analyze = st.button("🔍 Analyze", use_container_width=True)

with col2:
    clear = st.button("🧹 Clear", use_container_width=True)

if clear:
    st.experimental_rerun()

# ---------------------------
# ANALYSIS
# ---------------------------
if analyze:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        cleaned = clean_text(user_input)
        input_vec = vectorizer.transform([cleaned])

        prediction = model.predict(input_vec)[0]
        probability = model.predict_proba(input_vec)[0]

        confidence = max(probability) * 100

        st.markdown("---")
        st.markdown("## 📊 Analysis Result")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📝 Input Text")
            st.write(user_input)

        with col2:
            st.markdown("### 🧠 Prediction")

            if confidence < 55:
                st.warning(f"🤔 Uncertain ({confidence:.2f}%)")
            elif prediction == 1:
                st.success(f"✅ Real News ({confidence:.2f}%)")
            else:
                st.error(f"🚨 Fake News ({confidence:.2f}%)")
                st.warning("⚠️ Possible misinformation detected")

        # ---------------------------
        # CONFIDENCE BAR
        # ---------------------------
        st.markdown("### 📊 Confidence Score")
        st.progress(int(confidence))

        # ---------------------------
        # PROBABILITY BREAKDOWN
        # ---------------------------
        st.markdown("### 📈 Probability Breakdown")

        real_prob = probability[1] * 100
        fake_prob = probability[0] * 100

        col1, col2 = st.columns(2)
        col1.metric("🟢 Real Probability", f"{real_prob:.2f}%")
        col2.metric("🔴 Fake Probability", f"{fake_prob:.2f}%")

        # ---------------------------
        # INSIGHT
        # ---------------------------
        st.markdown("### 💡 Insight")

        if prediction == 0:
            st.info("This content contains patterns often associated with misinformation.")
        else:
            st.info("This content resembles credible and factual reporting.")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.caption("🚀 Built using Machine Learning & Streamlit")