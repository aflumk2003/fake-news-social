import os
import urllib.request
import pickle

MODEL_URL = "https://raw.githubusercontent.com/aflumk2003/fake-news-social/main/model.pkl"
VECTORIZER_URL = "https://raw.githubusercontent.com/aflumk2003/fake-news-social/main/vectorizer.pkl"

def download_file(url, filename):
    if not os.path.exists(filename):
        with open(filename, "wb") as f:
            f.write(urllib.request.urlopen(url).read())

download_file(MODEL_URL, "model.pkl")
download_file(VECTORIZER_URL, "vectorizer.pkl")

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))