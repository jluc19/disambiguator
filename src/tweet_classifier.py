import nltk
from nltk.corpus import stopwords
from nltk.util   import ngrams
import random

custom_stopwords = []
wordlist = []

# Opens the file with annotated tweet data, parses the tweets and labels to form
# a list of tuples of the form: ("This is a tweet", 3)
def parse_labeled_data(filename):

	f = open(filename, 'r')
	file_content = f.readlines()

	i = 1

	one = 0
	two = 0
	three = 0
	limit = 100

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
			elif l == 3 and three < limit:
				tweets_and_labels.append(elem)
				three = three + 1
		i = i + 1
	return tweets_and_labels

def split_feature_set(tweets):
	size = len(tweets)
	half = size / 2
	i = 0
	train = tweets[1:half]
	test = tweets[half:size]
	return (train, test)

# Given a tweet / string, returns a list of words
def tokenize_tweet(tweet):
	tokens = nltk.word_tokenize(tweet)

	# only take words - things with letters ONLY 
	words = [w for w in tokens if w.isalpha()]
	# only take english words
	# words = [w for w in words if not w in stopwords.words('english')]
	# only take USEFUL words
	words = [w for w in words if not w in custom_stopwords]

	return words

# Returns a featureset based on ngrams
# if n = 3, will return unigrams, bigrams and trigrams
def ngrams_features(corpus, n=1):
	words = []
	for (tweet, label) in corpus:
		gramsdict = {}
		for i in range(n):
			if i == 0 :
				grams = tokenize_tweet(tweet)
			else:
				grams = ngrams(tokenize_tweet(tweet), i + 1)
			for gr in grams:
				gramsdict[gr] = True
		words.append((gramsdict, label))
	return words

def feature_extractor(doc):
	docwords = set(doc)
	features = {}
	for i in docwords:
		features['contains(%s)' % i] = True
	return features

###		ACTUAL STUFF	###
filename = "../datatxt/parsed/labeled_tweets.txt"
tweets_and_labels = parse_labeled_data(filename)
random.shuffle(tweets_and_labels)

tweets = ngrams_features(tweets_and_labels, 2)

(train, test) = split_feature_set(tweets)

#Creates a training set - classifier learns distribution of true/falses in the input.
#training_set = nltk.classify.apply_features(feature_extractor, train)

# classifier
classifier = nltk.NaiveBayesClassifier.train(train)

print classifier.show_most_informative_features(n=100)

input = "got"
input = input.lower()
input = tokenize_tweet(input)
print input
features = feature_extractor(input)
print features
output = classifier.classify(features)
print 'got: ' + str(output)

prob = classifier.prob_classify(features)
print prob.prob(1)
print prob.prob(2)
print prob.prob(3)

#test_set = nltk.classify.apply_features(feature_extractor, test)
test_accuracy = nltk.classify.accuracy(classifier, test)

print 'test accuracy: ' + str(test_accuracy)



