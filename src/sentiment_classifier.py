#!/bin/python

import re
import csv
import sys
import math
import random
import itertools
import operator
from scipy.sparse import lil_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np

stops = [w.strip() for w in open('stopwords.txt').readlines()]

sentiments = ['strongly negative', 'negative', 'neutral', 'positive', 'strongly positive']
dv = DictVectorizer()
le = LabelEncoder()

#read in raw data from and return a list of (tweet, label, student name) tuples
def get_raw_data():
	data = []
	for line in open('all_data.txt').readlines():
		name, tweet, label = line.strip().split('\t')
		data.append((tweet, label, name))
	return data

#normalize the tweet tokens by lowercasing and removing stop words. 
#you can try not removing stopwords, too! maybe the stop words are useful sometimes...?
def normalize(tweet): return [t for t in tweet.lower().split() if not(t in stops)]

def ngrams(iterable, n=1):
	l = len(iterable)
	for idx in range(l):
		if idx + n < l : yield iterable[idx:idx+n]


#returns all n grams in toks
def ngram_features(toks, n=1) : 
	return {str(w) : 1 for w in ngrams(toks,n)}

#your feature which magically produces perfect accuracy
def your_feature_function(toks, sent_dick): 
	dick = {}
	for w in toks :
		if w in sent_dick:
			#dick[w] = 5
			if sent_dick[w] == 'positive':
				dick[w] = 1
			elif sent_dick[w] == 'neutral':
				dick[w] = 1
			elif sent_dick[w] == 'negative':
				dick[w] = 1
		else:
			dick[w] = 0

	return dick

#extract features from data and return a dictionary of {feature : value} pairs
def get_features(data, use_features):
	features = list()
	sentiment_dictionary = get_sentiment_dictionary('subjectivity_data/subjclueslen1-HLTEMNLP05.tff')
	
	for tweet in data:
		toks = normalize(tweet)
		tweet_features = {}
		#constant feature, provides no information
		if 'baseline' in use_features : tweet_features.update({'im a useless feature' : 1}) 
		#the length of the tweet. probably a useless feature as well.
		if 'length' in use_features : tweet_features.update({'length' : len(toks)}) 
		#raw words in the tweet
		if 'ngrams' in use_features : 
			tweet_features.update(ngram_features(toks))
		#TODO make it work
		if 'godfeature' in use_features : tweet_features.update(your_feature_function(toks, sentiment_dictionary))
		
		features.append(tweet_features)

	return dv.fit_transform(features)

#vectorize feature dictionaries and return feature and label matricies
def get_x_y(data, features=['baseline']):
	le.fit(sentiments)
	Y = le.transform([d[1] for d in data])
	X = get_features([d[0] for d in data], use_features=features)
	names = [d[2] for d in data]
	return Y, X, names

#train and multinomial naive bayes classifier
def train_classifier(X, Y):
	clf = MultinomialNB()
	clf.fit(X,Y)
	return clf 

#test the classifier
def test_classifier(clf, X, Y):
	return clf.score(X,Y)

#leave one out cross validation, where a different student's tweets serve as the test set in each iteration
def cross_validate_loo(X, Y, names):
	train_accs = []
	test_accs = []
	numfolds = len(set(names))
	for test_name in set(names) : 
		print 'Fold %s : '%(test_name),
		x_train = X[[i for i,n in enumerate(names) if n==test_name],:]
		y_train = [Y[i] for i,n in enumerate(names) if n==test_name]
		x_test = X[[i for i,n in enumerate(names) if not(n==test_name)],:]
		y_test = [Y[i] for i,n in enumerate(names) if not(n==test_name)]
		clf = train_classifier(x_train, y_train)
		train_acc = test_classifier(clf, x_train, y_train)
		train_accs.append(train_acc)
		acc = test_classifier(clf, x_test, y_test)
		test_accs.append(acc)
		print '%.05f'%(acc)
	train_average = float(sum(train_accs))/ numfolds
	average = float(sum(test_accs))/ numfolds
	print 'Train Average : %.05f'%(train_average)
	print 'Test Average : %.05f'%(average)
	return train_average, average

#normal cross validation	
def cross_validate(X, Y, numfolds=3, test_data=None):
	train_accs = []
	test_accs = []
	split = 1.0 / numfolds
	for i in range(numfolds):
		if test_data : x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_data)
		else : x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=split)
		clf = train_classifier(x_train, y_train)
		train_acc = test_classifier(clf, x_train, y_train)
		train_accs.append(train_acc)
		acc = test_classifier(clf, x_test, y_test)
		test_accs.append(acc)
		print 'Fold %d : %.05f'%(i,acc)
	train_average = float(sum(train_accs))/ numfolds
	average = float(sum(test_accs))/ numfolds
	print 'Train Average : %.05f'%(train_average)
	print 'Test Average : %.05f'%(average)
	return train_average, average
	
#returns a dictionary {sentiment : list of words} taken from manually constructed lists of sentiment words
def get_sentiment_dictionary(path):
	#format of lines in file : type=strongsubj len=1 word1=affable pos1=adj stemmed1=n priorpolarity=positive
	dictionary = {}
	for line in open(path).readlines():
		m = re.match('(.*) word1=((\w|-)*) (.*) priorpolarity=(\w*)', line.strip())
		_, word, _, _, sentiment = m.groups()
		if word not in dictionary : dictionary[word] = sentiment
		#if sentiment not in dictionary : dictionary[sentiment] = set()
	return dictionary

#trains classifier on increasing amounts of data and returns accuracy for each
def learning_curve(X, Y, blocks=11):
	accs = []
	blocksize = 1.0 / blocks
	testblock = int(X.shape[0]*blocksize)
	for i in range(blocks-1): 
		split = blocksize*(i+1)	
		if split == 1.0 : 
			acc = cross_validate(X, Y, test_data=testblock)
			accs.append((split, acc))
		else:
			x_ignore, x_use, y_ignore, y_use = train_test_split(X, Y, test_size=split)
			acc = cross_validate(x_use, y_use, test_data=testblock)
		accs.append((split, acc))
	for a in accs:	print a
	return accs

#draw a learning curve
def plot_learning_curve(accs):
	x = [split for split, (train, test) in accs]
	y_train = [train for split, (train, test) in accs]
	y_test = [test for split, (train, test) in accs]

	#table version for those who are visually impaired
	print '%data\taccuracy'
	for split,acc in zip(x,y_test) : print '%.02f\t%.03f'%(split,acc)

	#graph version
	plt.plot(x,y_test)
	plt.xlabel("% of data used to train")
	plt.ylabel("Error Rate")
	plt.show()

#successively adds each feature in feats, and plots the new accuracy with that feature added
def add_features():
	feats = ['baseline', 'length', 'ngrams']
	accs = {}
	for i in range(len(feats)) : 
		Y, X, names = get_x_y(get_raw_data(), features=feats[:i+1])
		accs[feats[i]] = cross_validate(X, Y)
	x = range(len(feats))
	test = [accs[f][1] for f in feats]
	train = [accs[f][0] for f in feats]

	#table version for those who are visually impaired
	print '\taccuracy'
	for a in accs: print '%s\t%.03f'%(a, accs[a][1])

	#graph version
	plt.bar(x, test, color='b', width=0.8)
	plt.ylabel("Accuracy")
	plt.xticks([xx+0.5 for xx in x], feats) 

	#uncomment these if you want to plot train/test accuracy side by side. this can help identify features that may be overfitting.
	#plt.bar(x, test, color='b', width=0.4)
	#plt.bar([xx+0.5 for xx in x], train, color='r', width=0.4)
	
	plt.show()

#successively removes features and gives the accuracy with that feature removed	
def ablation():
	feats = ['length', 'ngrams']
	accs = {}
	
	Y, X, names = get_x_y(get_raw_data(), features=['length', 'ngrams'])
	accs['all'] = cross_validate(X, Y)

	for i in range(len(feats)) : 
		Y, X, names = get_x_y(get_raw_data(), features=feats[:i]+feats[i+1:])
		accs['-'+feats[i]] = cross_validate(X, Y)
	labels = ['all'] + ['-'+f for f in feats]
	x = range(len(accs))
	test = [accs[f][1] for f in labels]

	#table version for those who are visually impaired
	print '\taccuracy'
	for a in labels : print '%s\t%.03f'%(a, accs[a][1])

	#graph version
	plt.bar(x, test, color='b', width=0.8)
	plt.ylabel("Accuracy")
	plt.xticks([xx+0.5 for xx in x], labels)
	plt.show()

#writes the top k features associated with each class to a file
def top_features(clf, k=20):
	feats = dict()
	for i,d in enumerate(dv.inverse_transform(clf.feature_log_prob_)): 
		label = le.inverse_transform([i])
		feats[label[0]] = sorted(d.iteritems(), key=operator.itemgetter(1), reverse=True)
	writer = csv.writer(open('top_features.txt', 'w'), delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	writer.writerow(['strongly negative', 'negative', 'neutral', 'positive', 'strongly positive'])
	for i in range(k):
		row = []
		row.append('%s %.02f'%feats['strongly negative'][i])
		row.append('%s %.02f'%feats['negative'][i])
		row.append('%s %.02f'%feats['neutral'][i])
		row.append('%s %.02f'%feats['positive'][i])
		row.append('%s %.02f'%feats['strongly positive'][i])
		writer.writerow(row)


if __name__ == '__main__' : 

	Y, X, names = get_x_y(get_raw_data(), features=['ngrams'])
	cross_validate(X,Y)
	#cross_validate_loo(X,Y,names)
	#add_features()
	#ablation()
	#plot_learning_curve(learning_curve(X,Y))
	top_features(train_classifier(X,Y))





