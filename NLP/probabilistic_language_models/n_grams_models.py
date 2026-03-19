import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

from collections import Counter

corpus = [
        "I love apple",
        "I love him",
        "I love NLP",
        "He loves apple",
        "You love me",
        "They love apple",
        "I love you and you love me"]

tokens = [word_tokenize(sentence.lower()) for sentence in corpus]

bigrams = []
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list, 2)))
    
bigrams_freq = Counter(bigrams)

trigrams = []
for token_list in tokens:
    trigrams.extend(list(ngrams(token_list, 3)))
    
trigram_freq = Counter(trigrams)

bigram = ("i", "love")  # target

prob_you = trigram_freq[("i", "love", "you")]/bigrams_freq[bigram]

print(f"The probability of the word you being present: {prob_you}")