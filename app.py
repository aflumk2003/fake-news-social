import os
import urllib.request
import pickle

# ---------------------------
# DOWNLOAD MODEL IF NOT EXISTS
# ---------------------------

MODEL_URL = "PASTE_MODEL_LINK"
VECTORIZER_URL = "PASTE_VECTORIZER_LINK"

if not os.path.exists("model.pkl"):
    urllib.request.urlretrieve(MODEL_URL, "model.pkl")

if not os.path.exists("vectorizer.pkl"):
    urllib.request.urlretrieve(VECTORIZER_URL, "vectorizer.pkl")

# ---------------------------
# LOAD
# ---------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))