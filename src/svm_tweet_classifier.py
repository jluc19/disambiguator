#Tweet classifier using SVM
#Boyang Zhang and Jason Lucibello

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
import random

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

def normalize(tweet): return [t for t in tweet.lower().split()]

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

	return dv.fit_transform(feat)

def get_x_y(data):
	le.fit(sentiments)
	#print data
	Y = le.transform([d[1] for d in data])
	X = get_features([d[0] for d in data])
	return Y, X

#random.shuffle

filename = "labeled_tweets.txt"
tweets_and_labels = parse_labeled_data(filename)
random.shuffle(tweets_and_labels)
Y, X = get_x_y(tweets_and_labels)
print Y
print X
clf = svm.SVC()
clf.fit(X, Y)
print clf

