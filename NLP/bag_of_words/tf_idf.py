import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "Dogs are very cute animals.",
    "She has very cute animals, one is a dog and the other is a bird.",
    "Cows produce milk."
    ]

tfidf_vectorizer = TfidfVectorizer()

# Convert text to numerical value
X = tfidf_vectorizer.fit_transform(documents)

feature_names = tfidf_vectorizer.get_feature_names_out()

vect_rep = X.toarray()
print(f"TF_IDF Vector Representation: {vect_rep}")

df_tfidf = pd.DataFrame(vect_rep, columns=feature_names)

# Average tf idf values
td_idf = df_tfidf.mean(axis=0)