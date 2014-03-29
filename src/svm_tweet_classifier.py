#Tweet classifier using SVM
#Boyang Zhang and Jason Lucibello

import nltk
import itertools

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm
import random
from sklearn.feature_selection import SelectFwe

sentiments = [1, 2, 3]
dv = DictVectorizer()
le = LabelEncoder()

def parse_labeled_data(filename):

	f = open(filename, 'r')
	file_content = f.readlines()

	i = 1

	one = 0
	two = 0
	three = 0
	limit = 200

	tweets_and_labels = []
	tweet = ''
	label = ''
	for line in file_content:
		line = line.rstrip('\n')
		if i % 2 == 1:
			tweet = line
		else:
			label = line
			l = int(label)
			elem = (tweet, l)
			if l == 1 and one < limit:
				tweets_and_labels.append(elem)
				one = one + 1
			elif l == 2 and two < limit:
				tweets_and_labels.append(elem)
				two = two + 1
			elif (l == 3 or l == 4) and three < limit:
				elem = (tweet, 3)
				tweets_and_labels.append(elem)
				three = three + 1
		i = i + 1
	print 'we got ' + str(one) + ' tweets labeled with a 1'
	print 'we got ' + str(two) + ' tweets labeled with a 2'
	print 'we got ' + str(three) + ' tweets labeled with a 3'
	return tweets_and_labels

def normalize(tweet): return [t for t in nltk.word_tokenize(tweet)] #use NLTK

def ngrams(iterable, n=1):
	l = len(iterable)
	for idx in range(l):
		if idx + n < l : yield iterable[idx:idx+n]

#returns all n grams in toks
def ngram_features(toks, n=1) : 
	n_dict = {}
	for i in range(1,n):
		n_dict.update({str(w) : 1 for w in ngrams(toks,i)})
	return n_dict

def get_features(data) :
	feat = []
	for tweet in data:
		toks = normalize(tweet)
		tweet_feat = ngram_features(toks, 2)
		feat.append(tweet_feat)

	feats = dv.fit_transform(feat)
	print dv.get_feature_names()
	return feats

def get_x_y(data):
	le.fit(sentiments)
	#print data
	Y = le.transform([d[1] for d in data])
	X = get_features([d[0] for d in data])
	return Y, X

#random.shuffle

filename = "labeled_tweets.txt"
tweets_and_labels = parse_labeled_data(filename)
#random.shuffle(tweets_and_labels)
Y, X = get_x_y(tweets_and_labels)

#splitting training and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#C = regularization parameter (keeps from overfitting): C is the degree of penalty (L1 or L2) (powers of 10)
#penalty sparse = l2 lowers angle so that no unigram can be super weighted, l1 removes features to shift the curve
#TODO: separate into train test eval

fs = SelectFwe(alpha=250.0)
print "Before", x_train.shape
x_train = fs.fit_transform(x_train, y_train)
print "After", x_train.shape
clf = svm.LinearSVC(C=.01, penalty = 'l2', dual=False)
clf.fit(x_train, y_train)
print "Training Accuracy"
print (classification_report(y_train, clf.predict(x_train)))

x_test = fs.transform(x_test)

print "Testing Accuracy"
print (classification_report(y_test, clf.predict(x_test)))








#output report of model success







#junk
#tw = ["I have diabetes"]
#imp = get_features(tw)
#print imp
#clf.predict([imp])

