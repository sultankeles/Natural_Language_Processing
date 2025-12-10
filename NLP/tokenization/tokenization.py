import nltk  # natural language toolkit

nltk.download("punkt")        # word & sentence tokenizer
nltk.download("punkt_tab")    # extra data needed by newer NLTK versions

text = "Welcome to the test section! This is a phrase used for testing. Please disregard its meaning."

# Word tokenization
word_tokens = nltk.word_tokenize(text)
print(word_tokens)


# Sentence tokenization
sentence_tokens = nltk.sent_tokenize(text)