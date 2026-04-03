import streamlit as st
import pickle
import re

# ---------------------------
# LOAD MODEL
# ---------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------------------
# CLEAN TEXT
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
    page_title="Fake News Detector",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------
# CUSTOM STYLING (🔥 UI BOOST)
# ---------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.title {
    font-size: 40px;
    font-weight: bold;
    color: #00ffd5;
}
.subtitle {
    font-size: 18px;
    color: #bbbbbb;
}
.box {
    padding: 20px;
    border-radius: 10px;
    background-color: #1c1f26;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# HEADER
# ---------------------------
st.markdown('<div class="title">📰 Fake News Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered verification of news & social media content</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------------------
# INPUT SECTION
# ---------------------------
st.markdown("### ✍️ Enter News Content")

user_input = st.text_area(
    "",
    placeholder="Paste WhatsApp forward, tweet, or headline here...",
    height=150
)

col1, col2 = st.columns([1,1])

with col1:
    analyze = st.button("🔍 Analyze", use_container_width=True)

with col2:
    clear = st.button("🧹 Clear", use_container_width=True)

if clear:
    st.experimental_rerun()

# ---------------------------
# PROCESSING
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

        # ---------------------------
        # RESULT SECTION
        # ---------------------------
        st.markdown("## 📊 Analysis Result")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📝 Input")
            st.write(user_input)

        with col2:
            st.markdown("### 🧠 Prediction")

            if confidence < 55:
                st.warning(f"🤔 Uncertain ({confidence:.2f}%)")
            elif prediction == 1:
                st.success(f"✅ Real News ({confidence:.2f}%)")
            else:
                st.error(f"🚨 Fake News ({confidence:.2f}%)")
                st.warning("⚠️ Potential misinformation detected")

        # ---------------------------
        # CONFIDENCE VISUAL
        # ---------------------------
        st.markdown("### 📊 Confidence Score")
        st.progress(int(confidence))

        # ---------------------------
        # PROBABILITY BREAKDOWN
        # ---------------------------
        st.markdown("### 📈 Probability Breakdown")

        real_prob = probability[1] * 100
        fake_prob = probability[0] * 100

        st.write(f"🟢 Real: {real_prob:.2f}%")
        st.write(f"🔴 Fake: {fake_prob:.2f}%")

        # ---------------------------
        # INSIGHT BOX
        # ---------------------------
        st.markdown("### 💡 Insight")

        if prediction == 0:
            st.info("This text contains patterns commonly seen in misinformation or sensational claims.")
        else:
            st.info("This text matches patterns typically found in reliable news reporting.")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.caption("Built using Machine Learning & NLP | Streamlit App")