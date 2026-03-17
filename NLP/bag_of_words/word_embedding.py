import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA    # principle component analysis: dimension reduction

from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess


sentences = [
            "My favorite animal is cat.",
            "Cats generally like to be independent.",
            "Dogs are friendly and loyal animals.",
            "Cats and dogs are domestic animals.",
            "Animals are humans' best friends."]

tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]

word2vec_model = Word2Vec(sentences = tokenized_sentences, vector_size = 50, window = 5, min_count = 1, sg = 0)

fast_text_model = FastText(sentences = tokenized_sentences, vector_size = 50, window = 5, min_count = 1, sg = 0)

# visualisation
def plot_word_embedding(model, title):
    
    word_vectors = model.wv
    
    words = list(word_vectors.index_to_key)[:1000]
    vectors = [word_vectors[word] for word in words]
    
    #PCA
    pca = PCA(n_components = 3)
    reduced_vectors = pca.fit_transform(vectors)
    
    # 3D visualisation
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(111, projection = "3d")

    # show vectors
    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2]) # x, y, z
    
    # tag the words
    for i, word in enumerate(words):
        ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2], word, fontsize = 12)
    
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    plt.show()
    
plot_word_embedding(word2vec_model, "Word2Vec")
plot_word_embedding(fast_text_model, "FastText")