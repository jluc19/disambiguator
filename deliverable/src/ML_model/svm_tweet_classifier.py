# -*- coding: utf-8 -*-
#Tweet classifier using SVM
#Boyang Zhang and Jason Lucibello

#Changes made based on this:https://www.youtube.com/watch?v=iFkRt3BCctg
#and this: http://nbviewer.ipython.org/github/herrfz/parallel_ml_tutorial/blob/master/notebooks/03%20-%20Text%20Feature%20Extraction%20for%20Classification%20and%20Clustering.ipynb

import matplotlib.pyplot as plt
import numpy as np
import re, random
from nltk import word_tokenize          # doctest: +SKIP
from nltk.stem import PorterStemmer, WordNetLemmatizer # doctest: +SKIP
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFwe
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.decomposition import RandomizedPCA
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle
np.set_printoptions(threshold='nan') #allows full printing of numpy.ndarrays

class LemmaTokenizer(object):
	def __init__(self):
		self.ps = PorterStemmer()
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		interim = [self.ps.stem(t) for t in word_tokenize(doc)]
		return [self.wnl.lemmatize(t) for t in interim]

def removeNonAscii(s): return "".join(i for i in s if ord(i)<128)

def output_ordered(ones, twos, threes):
		#Output tweets in order by category
		newFile = open('../training_data/ordered_tweets.txt', 'w')
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

def parse_labeled_data(filename):
	#variable setup
	ones, twos, threes, tweets_and_labels = ([] for i in range(4))
	tweet, label = '', ''
	i = 1
	with open(filename, 'r') as f:
		for line in f:
			if line.startswith('###'):
				continue
			line = line.rstrip('\n')
			removeNonAscii(line)
			if i % 2 == 1:
				line = re.sub("!","", line)
				line = re.sub("^\s+","", line)
				line = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''','',line)
				line = normalize(line)

				#Other Potential Levers
				#line = re.sub('@[^\s]+','USER',line)
				#line = re.sub(r'#([^\s]+)', r'\1', line)
				#line = re.sub(r"\b\d+\s?(?:[wW]|px|Px|[Pp]ixels|[hH])?\s*(?:x|by|X)\s*\d+\s?(?:[hH]|px|Px|[Pp]ixels|[wW])?\b",'',line)
				#line = re.sub(r"""((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.‌​][a-z]{2,4}/)(?:[^\s()<>]+|(([^\s()<>]+|(([^\s()<>]+)))*))+(?:(([^\s()<>]+|(‌​([^\s()<>]+)))*)|[^\s`!()[]{};:'".,<>?«»“”‘’]))""",'',line)
				#line = re.sub(r"[A-Za-z0-9\.\+_-]+@[A-Za-z0-9\._-]+\.[a-zA-Z]*",'',line)
				#line = re.sub(r"""\d+\s?(?:DPI|dpi)""",'',line)
				#line = re.sub(r"""#[A-Fa-f0-9]{6}""",'',line)
				
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

	#Determine duplicates
	#dup = open('duplicates.txt', 'w')
	#dup.write(str([x for x, y in collections.Counter(duplicates).items() if y > 1]))
	#output_ordered(ones, twos, threes)

	#remove duplicates
	ones = list(set(ones))
	twos = list(set(twos))
	threes = list(set(threes))

	smallest = min([len(l) for l in [ones, twos, threes]])
	print 'we have ' + str(len(ones)) + ' tweets labeled with a 1'
	print 'we have ' + str(len(twos)) + ' tweets labeled with a 2'
	print 'we have ' + str(len(threes)) + ' tweets labeled with a 3'
	print 'smallest list is of size' + str(smallest)

	#shuffling
	random.shuffle(ones)
	random.shuffle(twos)
	random.shuffle(threes)

	#trimming
	ones = ones[:smallest]
	twos = twos[:smallest]
	threes = threes[:smallest]

	#concatenating
	tweets_and_labels.extend(ones)
	tweets_and_labels.extend(twos)
	tweets_and_labels.extend(threes)

	random.shuffle(tweets_and_labels)
	return tweets_and_labels

def normalize(tweet): 
	# get rid of certain punctuation chars
	symbols_to_eliminate = ['.', '-', ',', '!', ]
	for symbol in symbols_to_eliminate:
		tweet.replace(symbol, '')
	return tweet

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
		feat.append(tweet)
	feats = dv.fit_transform(feat)
	return feats

def get_x_y(data):
	le.fit(sentiments)
	Y = le.transform([d[1] for d in data])
	X = get_features([d[0] for d in data])
	return Y, X

def print_top_features(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top20 = np.argsort(clf.coef_[i])[-20:]
        print("%s: %s" % (class_label, " ".join(feature_names[j] + "\n" for j in top20)))
        print("\n")

sentiments = [1 ,2, 3]
target_names = ["Self", "Another Person", "General Statement"]

dv = TfidfVectorizer(ngram_range=(1,2), min_df=0.0005, max_df=0.3, tokenizer=LemmaTokenizer())
le = LabelEncoder()

filename = "../training_data/labeled_data/diabetes_training_set.txt"
tweets_and_labels = parse_labeled_data(filename)

#random.shuffle(tweets_and_labels)
Y, X = get_x_y(tweets_and_labels)

#splitting training and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

print x_train.shape
print x_test.shape
print y_train.shape
print y_test.shape

fs = SelectFwe(alpha=150.0)

print "Before", x_train.shape, x_test.shape

sel = SelectPercentile(chi2, percentile=80)
sel.fit(x_train, y_train)

x_train = sel.transform(x_train)
x_test = sel.transform(x_test)

clf = svm.LinearSVC(C=10, penalty='l2', loss='l1', dual=True, fit_intercept=False, class_weight='auto')
x_train = fs.fit_transform(x_train, y_train)
clf.fit(x_train, y_train)

print "After", x_train.shape, x_test.shape

print "Training Accuracy"
print (classification_report(y_train, clf.predict(x_train), target_names=target_names))
x_test = fs.transform(x_test)
print "After", x_train.shape, x_test.shape
print "Testing Accuracy"
print clf.predict(x_test)
print (classification_report(y_test, clf.predict(x_test), target_names=target_names))

#print_top_features(dv, clf, target_names)

graph = False

if(graph):
	#scatter plot of scores
	train_small_pca = RandomizedPCA(n_components=3,).fit_transform(x_train)
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	for i, c in zip(np.unique(y_train), cycle(colors)):
	    plt.scatter(train_small_pca[y_train == i, 0],
	               train_small_pca[y_train == i, 1],
	               c=c, label=sentiments[i], alpha=0.5)   
	_ = plt.legend(loc='best')
	plt.show()

if(graph):
	#graphing setup
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
		if val == 0: val = 'b'; mark = 'o';
		elif val == 1: val = 'r'; mark = '+';
		else: val = 'g'; mark = '^';
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