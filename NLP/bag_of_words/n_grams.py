from sklearn.feature_extraction.text import CountVectorizer

documents = [
            "This labour is an example of NGram.",
            "This labour is an example of Natural Language Models."]

# unigram, bigram, trigram
vectorizer_unigram = CountVectorizer(ngram_range = (1,1))
vectorizer_bigram = CountVectorizer(ngram_range = (2,2))
vectorizer_trigram = CountVectorizer(ngram_range = (3,3))

X_unigram = vectorizer_unigram.fit_transform(documents)
unigram_features = vectorizer_unigram.get_feature_names_out()

X_bigram = vectorizer_bigram.fit_transform(documents)
bigram_features = vectorizer_bigram.get_feature_names_out()

X_trigram = vectorizer_trigram.fit_transform(documents)
trigram_features = vectorizer_trigram.get_feature_names_out()

print(f"Unigram Features: {unigram_features}")
print(f"Bigram Features: {bigram_features}")
print(f"Trigram Features: {trigram_features}")