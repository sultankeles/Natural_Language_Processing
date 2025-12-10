import nltk

from nltk.corpus import stopwords

nltk.download("stopwords")

# Stopwords analysis
# For English
stop_words_eng = set(stopwords.words("english"))

text = "There are some examples of handling stop words from some texts."
text_list = text.split()
filtered_words = [word for word in text_list if word.lower() not in stop_words_eng]
print(f"filtered_words: {filtered_words}")

# For Turkish
stop_words_tr = set(stopwords.words("turkish"))

metin = "Bu cümle durdurma kelimeleri tespit etmek için oluşturulmuş bir metindir."
metin_list = metin.split()

filtered_words_tr = [word for word in metin_list if word.lower() not in stop_words_tr]
print(f"filtered_words: {filtered_words_tr}")

# Without library

# Custom stopwords list
tr_stopwords = ["için", "bir", "bu", "ile", "mi", "mu", "şey", "özel"]

metin2 = "Bu bir denemedir. Amacımız bu metin içinde bulunan özel karakterleri mi elemek olacak, yoksa başka bir şey mi olacak?"

filtered_words_tr_custom = [word for word in metin2.split() if word.lower() not in tr_stopwords]
detected_stopwords = [word for word in metin2.split() if word.lower() in tr_stopwords]

print(f"filtered_words: {filtered_words_tr_custom}")