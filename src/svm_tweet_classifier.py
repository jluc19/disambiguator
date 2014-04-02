# -*- coding: utf-8 -*-
#Tweet classifier using SVM
#Boyang Zhang and Jason Lucibello

import nltk
import numpy as np
import itertools
from sklearn import svm, grid_search, datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm
import random
from sklearn.feature_selection import SelectFwe

sentiments = [1 ,2, 3]
target_names = ["Self", "Another Person", "General Statement"]
dv = DictVectorizer()
le = LabelEncoder()

def removeNonAscii(s): return "".join(i for i in s if ord(i)<128)
def parse_labeled_data(filename):

	f = open(filename, 'r')
	file_content = f.readlines()

	i = 1

	ones = []
	twos = []
	threes = []

	tweets_and_labels = []
	tweet = ''
	label = ''
	for line in file_content:
		if line.startswith('###'):
			continue
		removeNonAscii(line)
		#print line
		line = line.rstrip('\n')
		if i % 2 == 1:
			tweet = line
		else:
			label = line
			l = int(label)
			elem = (tweet, l)
			if l == 1:
				tweets_and_labels.append(elem)
				ones.append(elem)
			elif l == 2:
				tweets_and_labels.append(elem)
				twos.append(elem)
			elif (l == 3 or l == 4):
				elem = (tweet, 3)
				threes.append(elem)
		i = i + 1
	print 'we got ' + str(len(ones)) + ' tweets labeled with a 1'
	print 'we got ' + str(len(twos)) + ' tweets labeled with a 2'
	print 'we got ' + str(len(threes)) + ' tweets labeled with a 3'
	smallest = min([len(l) for l in [ones, twos, threes]])
	print 'smallest list is of size' + str(smallest)
	random.shuffle(ones)
	random.shuffle(twos)
	random.shuffle(threes)
	ones = ones[:smallest]
	twos = twos[:smallest]
	threes = threes[:smallest]
	tweets_and_labels = [] #we were GOD DAMN MISSING THIS LINE
	tweets_and_labels.extend(ones)
	tweets_and_labels.extend(twos)
	tweets_and_labels.extend(threes)
	print "ARRAY LENGTH", len(tweets_and_labels)
	random.shuffle(tweets_and_labels)
	return tweets_and_labels


	#THIS CODE NEEDS TO BE CLEANED UP

def normalize(tweet): 
	# get rid of certain punctuation chars
	symbols_to_eliminate = ['.', '-', ',']
	for symbol in symbols_to_eliminate:
		tweet.replace(symbol, '')

	toks = nltk.word_tokenize(tweet)

	# only take words - things with letters ONLY 
	# toks = [w for w in toks if w.isalpha()]

	return toks

def ngrams(iterable, n=1):
	l = len(iterable)
	for idx in range(l):
		if idx + n < l : yield iterable[idx:idx+n]

#returns all n grams in toks
def ngram_features(toks, n=1) : 
	n_dict = {}
	for i in range(1,n+1):
		n_dict.update({str(w) : 1 for w in ngrams(toks,i)})
	return n_dict

def get_features(data) :
	feat = []
	for tweet in data:
		toks = normalize(tweet)
		tweet_feat = ngram_features(toks, 2)
		feat.append(tweet_feat)
	feats = dv.fit_transform(feat)
	return feats

def get_x_y(data):
	le.fit(sentiments)
	#print data
	Y = le.transform([d[1] for d in data])
	X = get_features([d[0] for d in data])
	print "Y, X SIZE", len(Y)
	return Y, X

def print_top_features(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top20 = np.argsort(clf.coef_[i])[-20:]
        print("%s: %s" % (class_label, " ".join(feature_names[j] for j in top20)))
        print("\n")

#random.shuffle

filename = "new_labeled_tweets.txt"
tweets_and_labels = parse_labeled_data(filename)
#random.shuffle(tweets_and_labels)
Y, X = get_x_y(tweets_and_labels)

#splitting training and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

#C = regularization parameter (keeps from overfitting): C is the degree of penalty (L1 or L2) (powers of 10)
#penalty sparse = l2 lowers angle so that no unigram can be super weighted, l1 removes features to shift the curve
#TODO: separate into train test eval

fs = SelectFwe(alpha=140.0)
print "Before", x_train.shape
x_train = fs.fit_transform(x_train, y_train)
print "After", x_train.shape
clf = svm.LinearSVC(C=10, penalty = 'l2', dual=False)
clf.fit(x_train, y_train)

print_top_features(dv, clf, target_names)

print "Training Accuracy"
print (classification_report(y_train, clf.predict(x_train), target_names=target_names))
x_test = fs.transform(x_test)

print "Testing Accuracy"
print (classification_report(y_test, clf.predict(x_test), target_names=target_names))


