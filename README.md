<div align="center">

```
███████╗ █████╗ ██╗  ██╗███████╗    ███╗   ██╗███████╗██╗    ██╗███████╗
██╔════╝██╔══██╗██║ ██╔╝██╔════╝    ████╗  ██║██╔════╝██║    ██║██╔════╝
█████╗  ███████║█████╔╝ █████╗      ██╔██╗ ██║█████╗  ██║ █╗ ██║███████╗
██╔══╝  ██╔══██║██╔═██╗ ██╔══╝      ██║╚██╗██║██╔══╝  ██║███╗██║╚════██║
██║     ██║  ██║██║  ██╗███████╗    ██║ ╚████║███████╗╚███╔███╔╝███████║
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═══╝╚══════╝ ╚══╝╚══╝ ╚══════╝
                                                                            
██████╗ ███████╗████████╗███████╗ ██████╗████████╗ ██████╗ ██████╗         
██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗        
██║  ██║█████╗     ██║   █████╗  ██║        ██║   ██║   ██║██████╔╝        
██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   ██║   ██║██╔══██╗        
██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║        
╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝        
```

**AI-Powered Misinformation Intelligence System**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.7%25-00D26A?style=flat-square)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-blueviolet?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)](CONTRIBUTING.md)

*Classify news articles as Real or Fake in milliseconds using TF-IDF + Logistic Regression*

[**🚀 Live Demo**](https://your-demo-link.streamlit.app) · [**📊 Dataset**](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) · [**📖 Docs**](#-documentation) · [**🐛 Report Bug**](https://github.com/aflumk2003/issues) · [**✨ Request Feature**](https://github.com/aflumk2003/issues)

</div>

---

## ⚡ What Is This?

> *In an era where a single viral post can shift public opinion within hours, the ability to instantly verify information is no longer a luxury — it's a necessity.*

**Fake News Detector** is an ML-powered classification engine that analyzes the linguistic fingerprint of any news article or social media post and determines its credibility with a probabilistic confidence score.

```
Input Text ──► Preprocessing ──► TF-IDF Vectorization ──► Logistic Regression ──► Verdict + Confidence
```

---

## 🖥️ Demo

<div align="center">

| Input | Verdict | Confidence |
|:------|:-------:|:----------:|
| *"Scientists confirm water found on Mars..."* | ✅ **REAL** | 94.2% |
| *"Government putting 5G chips in vaccines..."* | 🚨 **FAKE** | 97.8% |
| *"New policy changes announced for..."* | 🤔 **UNCERTAIN** | 51.3% |

</div>

> 📸 Add screenshots of your running app here — visuals dramatically improve GitHub engagement.

---

## ✨ Features

```
┌─────────────────────────────────────────────────────────────┐
│                    CORE CAPABILITIES                         │
├──────────────────────┬──────────────────────────────────────┤
│  🔍 Text Analysis    │  Real-time classification of any     │
│                      │  news article or social media post   │
├──────────────────────┼──────────────────────────────────────┤
│  📊 Confidence Score │  Probabilistic output with           │
│                      │  Real vs Fake percentage breakdown   │
├──────────────────────┼──────────────────────────────────────┤
│  ⚠️  Uncertainty Mode │  Flags low-confidence predictions   │
│                      │  instead of forcing a binary result  │
├──────────────────────┼──────────────────────────────────────┤
│  🎨 Modern UI        │  Clean, responsive Streamlit         │
│                      │  interface with chart visualizations │
└──────────────────────┴──────────────────────────────────────┘
```

---

## 🧠 Model Architecture

```
                        ╔══════════════════╗
  Raw Text Input        ║   PREPROCESSING  ║
  ─────────────► Text ──║ • Lowercasing    ║──►  Cleaned Text
                        ║ • Noise Removal  ║
                        ║ • Normalization  ║
                        ╚══════════════════╝
                                 │
                                 ▼
                        ╔══════════════════╗
                        ║  TF-IDF ENGINE   ║
                        ║                  ║
                        ║ • Unigrams       ║──►  Feature Matrix
                        ║ • Bigrams        ║     [n × vocab_size]
                        ║ • Trigrams       ║
                        ╚══════════════════╝
                                 │
                                 ▼
                        ╔══════════════════╗
                        ║ LOGISTIC REGR.   ║
                        ║                  ║
                        ║ P(Fake|X)  ──────║──►  🚨 FAKE  (>0.65)
                        ║ P(Real|X)  ──────║──►  ✅ REAL  (<0.35)
                        ║                  ║──►  🤔 UNCERTAIN
                        ╚══════════════════╝
```

| Parameter | Value |
|:----------|:------|
| Algorithm | Logistic Regression (L2 regularization) |
| Vectorizer | TF-IDF with n-gram range (1, 3) |
| Max Features | 50,000 |
| Confidence Threshold | 65% |
| Training Accuracy | ~99.1% |
| Validation Accuracy | ~98.7% |

---

## 📂 Project Structure

```
fake-news-detector/
│
├── 📱 app.py                 # Streamlit web application
├── 🏋️ train.py               # Model training pipeline
│
├── 🧠 model.pkl              # Serialized trained model
├── 🔤 vectorizer.pkl         # Fitted TF-IDF vectorizer
│
├── 📊 dataset/
│   ├── True.csv              # 21,417 verified real news articles
│   └── Fake.csv              # 23,481 labeled fake news articles
│
├── 📋 requirements.txt       # Python dependencies
└── 📖 README.md              # You are here
```

---

## 📊 Dataset

This project is trained on the **Kaggle Fake and Real News Dataset** — one of the most widely cited misinformation research datasets.

| Split | Articles | Source |
|:------|:--------:|:-------|
| Real News | **21,417** | Reuters |
| Fake News | **23,481** | PolitiFact / unreliable sites |
| **Total** | **44,898** | Mixed |

> 🔗 **Download:** [kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## ⚙️ Setup & Installation

### Prerequisites

```bash
Python 3.9+  |  pip  |  Git
```

### 1 — Clone

```bash
git clone https://github.com/aflumk2003/fake-news-detector.git
cd fake-news-detector
```

### 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### 3 — Prepare Dataset

Download the Kaggle dataset and place in project root:

```
fake-news-detector/
├── True.csv        ← place here
├── Fake.csv        ← place here
└── ...
```

### 4 — Train the Model

```bash
python train.py
```

> ⏱️ Training takes ~30–60 seconds on a standard laptop. Two files will be generated: `model.pkl` and `vectorizer.pkl`

### 5 — Launch the App

```bash
python -m streamlit run app.py
```

Open your browser at **`http://localhost:8501`**

---

## 🧪 How It Works

```
Step 1: INPUT
  └─ User pastes a news article or social media text

Step 2: PREPROCESSING
  └─ Lowercase → strip HTML/URLs → remove punctuation → normalize whitespace

Step 3: VECTORIZATION
  └─ TF-IDF transforms text into a high-dimensional sparse feature vector
     Each feature = weighted importance of a word/phrase in the corpus

Step 4: CLASSIFICATION
  └─ Logistic Regression outputs P(Fake) and P(Real)
     Confidence < 65% → flagged as "Uncertain"

Step 5: DISPLAY
  └─ Verdict card + probability bar chart rendered in Streamlit
```

---

## ⚠️ Limitations

| Limitation | Details |
|:-----------|:--------|
| **Domain Bias** | Trained primarily on US political news; may underperform on tech, sports, or Indian news |
| **Context Blindness** | Cannot understand satire, sarcasm, or cultural irony |
| **Static Knowledge** | Model doesn't update with new events after training |
| **Short Texts** | Social media posts < 20 words yield lower confidence |

---

## 🚀 Roadmap

- [ ] 🔥 **BERT / RoBERTa** — Transformer-based classification for contextual understanding
- [ ] 🌐 **News API Integration** — Paste a URL, auto-fetch and analyze full articles
- [ ] 🧠 **Explainable AI** — Highlight words that triggered the Fake/Real decision
- [ ] 📊 **Analytics Dashboard** — Track and visualize detection history
- [ ] 🌍 **Multilingual Support** — Expand beyond English
- [ ] 🔁 **Active Learning** — Improve model from user feedback

---

## 🛠️ Tech Stack

| Layer | Technology |
|:------|:-----------|
| Language | Python 3.9+ |
| ML Framework | Scikit-learn |
| NLP | TF-IDF (sklearn) |
| Data | Pandas, NumPy |
| UI | Streamlit |
| Serialization | Pickle |

---

## 🤝 Contributing

Contributions are welcome. Please follow the standard fork → branch → PR workflow.

```bash
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
```

Open a pull request and describe what you changed and why.

---

## 📄 License

This project is released under the **MIT License** — free for educational and personal use.

---

<div align="center">

**Built by [Fidel M](https://github.com/aflumk2003)**

*If this project helped you, consider giving it a ⭐*

</div>
