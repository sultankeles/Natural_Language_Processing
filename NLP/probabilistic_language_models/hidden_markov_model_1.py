import nltk
from nltk.tag import hmm

train_data = [
            [("I", "PRP"), ("am", "VBP"), ("a", "DT"), ("student", "NN")],
            [("You", "PRP"), ("are", "VBP"), ("a", "DT"), ("teacher", "NN")]
            ]

# train HMM
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

test_sentence = "I am a teacher".split()

tags = hmm_tagger.tag(test_sentence)

print(f"New Sntence: {tags}")