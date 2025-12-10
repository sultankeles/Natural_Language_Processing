import nltk

nltk.download("wordnet")    # for lemmatization

from nltk.stem import PorterStemmer

# create object Porter Stemmer
stemmer = PorterStemmer()

words = ["running", "runner", "ran", "runs", "better", "go", "went"]

new_words = ["feeling", "feel", "felt", "feels", "get", "got", "gotten", "take", "took", "give", "given", "gave", "lose", "lost", "rise", "rose", "risen"]

# finding stem of words w/ using stem() function
stems = [stemmer.stem(w) for w in words]
stems_2 = [stemmer.stem(w) for w in new_words]

print(f"Stem: {stems}")
print(f"Stem: {stems_2}")

# lemmatizations
from nltk.stem import WordNetLemmatizer

lemmetizer = WordNetLemmatizer()

lemmas = [lemmetizer.lemmatize(w, pos="v") for w in words]  # position = "verb"
lemmas_2 = [lemmetizer.lemmatize(w, pos="v") for w in new_words]

print(f"Lemmas: {lemmas}")
print(f"Lemmas: {lemmas_2}")