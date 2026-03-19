from nltk.classify import MaxentClassifier

train_data = [
            ({"love":True, "amazing":True, "happy":True, "terrible":False}, "positive"),
            ({"hate":True, "terrible":True}, "negative"),
            ({"joy":True, "happy":True, "hate":False}, "positive"),
            ({"sad":True, "depressed":True, "love":False}, "negative")
            ]

classifier = MaxentClassifier.train(train_data, max_iter = 10)

test_sentence = "I love this movie and it was amazing!"
features = {word: (word in test_sentence.lower().split()) for word in ["love", "amazing", "terrible", "happy", "joy", "depressed", "sad"]}

label = classifier.classify(features)
print(f"Result: {label}")