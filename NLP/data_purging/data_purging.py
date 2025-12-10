# Elimination of excess spaces in texts

text = "Hello      World!"

'''
text.split()
Out[2]: ['Hello', 'World!']
'''

cleaned_text = " ".join(text.split())
print(f"text: {text} \n cleaned_text: {cleaned_text}")


# Case conversion

converted_text = cleaned_text.lower()
print(f"cleaned_text: {cleaned_text} \n converted_text: {converted_text}")


# Remove punctuations
import string

n_text = "Agnes Gonxha Bojaxhiu (popularly known as ‘Mother Teresa’) was born on August 26, 1910."
removed_punctuations = n_text.translate(str.maketrans("", "", string.punctuation))
print(f"new text: {n_text} \n removed punctuations: {removed_punctuations}")


# Remove special characters (%, @, /, *, #)
import re

text2 = "This text contains special characters like %, @, /, , and # that need to be removed."
rsc = re.sub(r"[^A-Za-z0-9\s]", "", text2)
print(f"text: {text2} \n removed special characters: {rsc}")


# Spell Correction
from textblob import TextBlob 

text3 = "Llet's finnd and fix the spalling miustakess."
corrected_text = TextBlob(text3).correct() # correct: fixes the spelling mistakes
print(f"text: {text3} \n correct versions: {corrected_text}")


# Remove html or url tags
from bs4 import BeautifulSoup

'''
Parse the HTML structure with Beautiful Soup, get the text part with the get_text tag
'''

html_text = "<div>Hello World!</div>" # incloud html tag
removed_html = BeautifulSoup(html_text, "html.parser").get_text()
print(f"html_text: {html_text} \n removed html: {removed_html}")