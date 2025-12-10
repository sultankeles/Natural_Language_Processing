# import libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

# import dataset
df = pd.read_csv("IMDB Dataset.csv")

documents = df["review"]
labels = df["sentiment"]    # positive or negative

# stopword analysis
stopwords = set(stopwords.words("english"))

# text cleaning
def clean_text(text):
    
    # Case conversion
    text = text.lower()

    # Removal of numbers
    text = re.sub(r"\d+", "", text)

    # Removal of special characters
    text = re.sub(r"[^\w\s]", "", text)

    # Removal of short words
    text = " ".join([word for word in text.split() if len(word) > 2])
    
    # Filter stopwords
    text = [word for word in text.split() if word not in stopwords]

    # Return cleaned text
    return " ".join(text)

cleaned_doc = [clean_text(row) for row in documents]

# bow
# describe vectorizer
vectorizer = CountVectorizer()

# convert text to numeric value (first 75 reviews)
X = vectorizer.fit_transform(cleaned_doc[:75])

# show word set
feature_names = vectorizer.get_feature_names_out()

# show vector representation
vector_rep = X.toarray()
print(f"Vector Representation: {vector_rep}")

df_bow = pd.DataFrame(vector_rep, columns = feature_names)

# show word frequencies
word_counts = X.sum(axis = 0).A1
word_freq = dict(zip(feature_names, word_counts))

# print first 5 words
mc_words = Counter(word_freq).most_common(5)
print(f"Most Common Five Words: {mc_words}")