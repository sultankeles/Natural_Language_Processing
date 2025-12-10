from sklearn.feature_extraction.text import CountVectorizer

documents = [
             "This is an example sentence.", 
             "And this one is another example."
             ]
             
vectorizer = CountVectorizer()

# convert text to numeric vectors
X = vectorizer.fit_transform(documents)

feature_names = vectorizer.get_feature_names_out()
print(f"word_cluster: {feature_names}")

vector_representation = X.toarray()
print(f"vector_representation: {vector_representation}")

'''
word_cluster:           'an' 'and' 'another' 'example' 'is' 'one' 'sentence' 'this']
vector_representation: [ 1     0       0         1      1     0       1        1   ]
                       [ 0     1       1         1      1     1       0        1   ]

'''