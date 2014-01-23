import nltk
#from text.classifiers import NaiveBayesClassifier
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import random
import string

def remove_non_ascii_1(text):
    return ''.join(i for i in text if ord(i)<128)

def parse_file(filename) :
	f = open(filename, 'r')
	file_contents = f.readlines()

	label_tweet = []

	for lines in file_contents:
		f_line = remove_non_ascii_1(lines)
		if(lines.split()[1] == 'health' or lines.split()[1] == 'sick'):
			curr_line = (" ".join(f_line.split()[2:]), 'pos')
		else:
			curr_line = (" ".join(f_line.split()[2:]), 'neg')
		label_tweet.append(curr_line)
	random.shuffle(label_tweet)
	return set(label_tweet)


def bigram_word_feats(words, score_fn=BigramAssocMeasures.likelihood_ratio, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

def unigram(tweet):
	return dict([(ngram,True) for ngram in nltk.word_tokenize(tweet)]) 

label_tweet = parse_file("annotated_tweets.txt")



def stripPunct(s):
	x = s.translate(string.maketrans("",""), string.punctuation)
	return x

def splitFeatureSet(featureset):
	pos = 0
	neg = 0

	train_set = []
	test_set = []
	for(d, c) in featureset:
		if c == 'pos' and pos <= 1000:
			pos+=1
			train_set.append((d,c))
		elif c == 'neg' and neg <= 1000:
			neg+=1
			train_set.append((d,c))
		else:
			test_set.append((d,c))

	return test_set, train_set


#bigrams
def bigrams():
	featureset = [(bigram_word_feats(nltk.word_tokenize(stripPunct(d))), c) for (d,c) in label_tweet]
	
	test_set, train_set = splitFeatureSet(featureset)

	classifier = nltk.NaiveBayesClassifier.train(train_set)
	print classifier.show_most_informative_features(100)
	print 'Test accuracy: ' + str(nltk.classify.accuracy(classifier, test_set))

#unigrams
def unigrams():
	featureset = [(unigram(stripPunct(tweet)), label) for tweet, label in label_tweet]
	

	test_set, train_set = splitFeatureSet(featureset)
	classifier = nltk.NaiveBayesClassifier.train(train_set)
	print classifier.show_most_informative_features(100)
	print 'Test accuracy: ' + str(nltk.classify.accuracy(classifier, test_set))

#maxEnt
def maxEnt():
	featureset = [(bigram_word_feats(nltk.word_tokenize(d)), c) for (d,c) in label_tweet]
	classifier = nltk.classify.maxent.BinaryMaxentFeatureEncoding(['pos', 'neg'], featureset)
	print classifier.show_most_informative_features(100)



bigrams()
unigrams()
#maxEnt()

#train_set = []
#for tweet, label in label_tweet:
#	model = nltk.NgramModel(2, nltk.word_tokenize(tweet))
#	train_set.append((model._ngrams, label))
	#for gram in model._ngrams:
		#rain_set.append((gram, label))
 