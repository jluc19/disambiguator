import nltk
from nltk.corpus import stopwords
from nltk.util   import ngrams
import random
import math

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

def split_feature_set(tweets):
	size = len(tweets)
	split = int(math.ceil(0.8 * size))
	train = tweets[1:split]
	test = tweets[split:]
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

# 	Returns a featureset based on ngrams
# 	if n = 3, will return unigrams, bigrams and trigrams
#	INPUT  : 1 tweet (string)
# 	OUTPUT : dictionary. key = gram, value = True
def ngrams_features(tweet, n=1):
	gramsdict = {}
	tweet_as_tokens = tokenize_tweet(tweet)
	for i in range(n):
		if i == 0 :
			grams = tweet_as_tokens
		else:
			grams = ngrams(tweet_as_tokens, i + 1)
		for gr in grams:
			gramsdict[gr] = True
	return gramsdict



###		ACTUAL STUFF	###
filename = "../datatxt/parsed/labeled_tweets.txt"
tweets_and_labels = parse_labeled_data(filename)
random.shuffle(tweets_and_labels)

features_for_tweets = [(ngrams_features(t, 2), l) for (t, l) in tweets_and_labels]

(train, test) = split_feature_set(features_for_tweets)

# classifier
classifier = nltk.NaiveBayesClassifier.train(train)

print classifier.show_most_informative_features(n=100)

input = 'i just got diabetes the other day'
input = input.lower()
input_list = tokenize_tweet(input)
print 'input is ' + str(input_list)
features = ngrams_features(input, 2)
print features
output = classifier.classify(features)
print 'output: ' + str(output)

prob = classifier.prob_classify(features)
print prob.prob(1)
print prob.prob(2)
print prob.prob(3)

test_accuracy = nltk.classify.accuracy(classifier, test)

print 'test accuracy: ' + str(test_accuracy)



