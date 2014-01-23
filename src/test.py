import pickle

f= open('my_classifier.pickle')
classifier = pickle.load(f)
f.close()

classifier.show_informative_features(5)
print classifier.classify("I am sooo sick")
print classifier.accuracy(test_set)