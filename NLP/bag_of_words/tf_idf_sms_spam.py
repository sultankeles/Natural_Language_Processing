import pandas as pd
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("sms_spam.csv")

# data purging
documents = df["text"]
labels = df["type"]

stopwords = set(stopwords.words("english"))

def clean_text(text):

    text = text.lower()

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"[^\w\s]", "", text)

    text = " ".join([word for word in text.split() if len(word) > 2])
    
    text = [word for word in text.split() if word not in stopwords]

    return " ".join(text)

cleaned_doc = [clean_text(row) for row in documents]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_doc)

feature_names = vectorizer.get_feature_names_out()
tfidf_score = X.mean(axis=0).A1

df_tfidf = pd.DataFrame({"word":feature_names, "tfidf_score":tfidf_score})

df_tfidf_sorted = df_tfidf.sort_values(by="tfidf_score", ascending=False)
print(df_tfidf_sorted.head(10))