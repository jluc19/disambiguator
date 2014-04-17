# -*- coding: utf-8 -*-
#Tweet classifier using SVM
#Boyang Zhang and Jason Lucibello

import nltk
import numpy as np
from numpy import exp,arange
from sklearn import svm, grid_search, datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFwe
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm, mlab

import random, re, collections, itertools

sentiments = [1 ,2, 3]
target_names = ["Self", "Another Person", "General Statement"]
dv = DictVectorizer()
le = LabelEncoder()

def removeNonAscii(s): return "".join(i for i in s if ord(i)<128)

def parse_labeled_data(filename):
	#variable setup
	ones, twos, threes, tweets_and_labels = ([] for i in range(4))
	tweet, label = '', ''
	i = 1
	newFile = open('../training_data/ordered_tweets.txt', 'w')
	dup = open('duplicates.txt', 'w')
	with open(filename, 'r') as f:
		for line in f:
			if line.startswith('###'):
				continue
			line = line.rstrip('\n')
			#removeNonAscii(line)
			#print line
			if i % 2 == 1:
				line = re.sub('@[^\s]+','USER',line)
				line = re.sub("^\s+","", line)
				line = re.sub(r'#([^\s]+)', r'\1', line)
				#line = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''','',line)
				tweet = line
			else:
				l = int(line)
				if l == 1:
					ones.append((tweet, l))
				elif l == 2:
					twos.append((tweet, l))
				else:
					threes.append((tweet, 3))
			i = i + 1


	duplicates = []
	duplicates.extend(ones)
	duplicates.extend(twos)
	duplicates.extend(threes)

	dup.write(str([x for x, y in collections.Counter(duplicates).items() if y > 1]))

	#remove duplicates
	ones = list(set(ones))
	twos = list(set(twos))
	threes = list(set(threes))

	for item, val in ones:
		newFile.write(item + "\n")
		newFile.write(str(val) + "\n")
	for item, val in twos:
		newFile.write(item + "\n")
		newFile.write(str(val) + "\n")
	for item, val in threes:
		newFile.write(item + "\n")
		newFile.write(str(val) + "\n")
	newFile.close()

	smallest = min([len(l) for l in [ones, twos, threes]])
	print 'we have ' + str(len(ones)) + ' tweets labeled with a 1'
	print 'we have ' + str(len(twos)) + ' tweets labeled with a 2'
	print 'we have ' + str(len(threes)) + ' tweets labeled with a 3'
	print 'smallest list is of size' + str(smallest)

	#shuffling
	#random.shuffle(ones)
	#random.shuffle(twos)
	#random.shuffle(threes)

	#trimming
	ones = ones[:smallest]
	twos = twos[:smallest]
	threes = threes[:smallest]

	#concatenating
	tweets_and_labels.extend(ones)
	tweets_and_labels.extend(twos)
	tweets_and_labels.extend(threes)

	#random.shuffle(tweets_and_labels)
	return tweets_and_labels

def normalize(tweet): 
	# get rid of certain punctuation chars
	symbols_to_eliminate = ['.', '-', ',']
	for symbol in symbols_to_eliminate:
		tweet.replace(symbol, '')

	toks = nltk.word_tokenize(tweet)

	# only take words - things with lowercase letters 
	toks = [w.lower() for w in toks]

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
		#print toks
		tweet_feat = ngram_features(toks, 2)
		feat.append(tweet_feat)
	feats = dv.fit_transform(feat)
	return feats

def get_x_y(data):
	le.fit(sentiments)
	#print data
	Y = le.transform([d[1] for d in data])
	X = get_features([d[0] for d in data])
	#print "Y, X SIZE", len(Y)
	return Y, X

def print_top_features(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top20 = np.argsort(clf.coef_[i])[-20:]
        print("%s: %s" % (class_label, " ".join(feature_names[j] for j in top20)))
        print("\n")

filename = "../training_data/ordered_tweets_no_duplicates.txt"
tweets_and_labels = parse_labeled_data(filename)
#random.shuffle(tweets_and_labels)
Y, X = get_x_y(tweets_and_labels)

#splitting training and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

#C = regularization parameter (keeps from overfitting): C is the degree of penalty (L1 or L2) (powers of 10)
#penalty sparse = l2 lowers angle so that no unigram can be super weighted, l1 removes features to shift the curve
#TODO: separate into train test eval

fs = SelectFwe(alpha=150.0)

print "Before", x_train.shape

clf = svm.LinearSVC(C=100, penalty='l2', dual = False)
clf.fit(x_train, y_train)




print "NO FEATURE SELECTION"
print "Training Accuracy"
#print clf.decision_function(x_train)
print (classification_report(y_train, clf.predict(x_train), target_names=target_names))
print "Testing Accuracy"
print (classification_report(y_test, clf.predict(x_test), target_names=target_names))





x_train = fs.fit_transform(x_train, y_train)
clf.fit(x_train, y_train)

print "After", x_train.shape





print "Training Accuracy"
print (classification_report(y_train, clf.predict(x_train), target_names=target_names))
x_test = fs.transform(x_test)
print "Testing Accuracy"
print (classification_report(y_test, clf.predict(x_test), target_names=target_names))



print_top_features(dv, clf, target_names)





decisions = clf.decision_function(x_test)
X = np.array(decisions[:,0]) #Self
Y = np.array(decisions[:,1]) #Other Person
Z = np.array(decisions[:,2]) #General Statements
points = []
for i, val in enumerate(X):
	points.append((X[i], Y[i], Z[i]))
points = list(set(points))
new_y = []
for i, val in enumerate(y_test):
	if val == 0:
		val = 'b'
		mark = 'o'
	elif val == 1:
		val = 'r'
		mark = '+'
	else:
		val = 'g'
		mark = '^'
	new_y.append((val, mark))

#3-D Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, val in enumerate(np.array(X)):
	ax.scatter3D(X[i], Y[i], Z[i], c=new_y[i][0], marker=new_y[i][1])

ax.set_xlabel('Self')
ax.set_ylabel('Another Person')
ax.set_zlabel('General Disease')
ax.set_autoscale_on(True)
plt.show()

#2-D Plot
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
for i, val in enumerate(np.array(X)):
	if new_y[i][0] != 'g': 
		ax2.scatter(X[i], Y[i], c=new_y[i][0], marker=new_y[i][1])
ax2.set_xlabel('Self')
ax2.set_ylabel('Another Person')
ax2.set_autoscale_on(True)
plt.show()




