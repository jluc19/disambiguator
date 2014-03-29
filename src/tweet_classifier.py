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
	# get rid of certain punctuation chars
	symbols_to_eliminate = ['.', '-', ',']
	for symbol in symbols_to_eliminate:
		tweet.replace(symbol, '')

	tokens = nltk.word_tokenize(tweet)

	# only take words - things with letters ONLY 
	words = [w for w in tokens if w.isalpha()]

	# only take USEFUL words
	words = [w for w in words if not w in custom_stopwords]

	return words

# 	Returns a featureset based on ngrams
# 	if n = 3, will return unigrams, bigrams and trigrams
#	INPUT  : 1 tweet (string)
# 	OUTPUT : dictionary. key = gram, value = True
def ngrams_features(tweet, frequency_dictionary, n=1, min_freq=6):
	gram_features = {}
	tweet_as_tokens = tokenize_tweet(tweet)
	for i in range(n):
		if i == 0 :
			grams = tweet_as_tokens
		else:
			grams = ngrams(tweet_as_tokens, i + 1)
		for gr in grams:
			if gr in frequency_dictionary:
			    if frequency_dictionary[gr] >= min_freq:
			        gram_features[gr] = True
	return gram_features

def get_ngram_frequencies(tweets_and_labels, n=1):
    d = {}
    for (t, l) in tweets_and_labels:
    	tokens = tokenize_tweet(t)
    	for i in range(n):
    		if i == 0:
    			grams = tokens
    		else:
    		    grams = ngrams(tokens, i + 1)
    		for gr in grams:
    		    if gr in d:
    		    	num = d[gr]
    		    	d[gr] = num + 1
    		    else:
    		    	d[gr] = 1
    return d



###		ACTUAL STUFF	###
# 1. Open labeled data, parse into tuples
n = 2
# min_freq = 3
filename = "../datatxt/parsed/labeled_tweets.txt"
tweets_and_labels = parse_labeled_data(filename)
random.shuffle(tweets_and_labels)

# 2. Create frequency dictionary of all words and tuples
frequency_dictionary = get_ngram_frequencies(tweets_and_labels, n)

# 3. get features out of tweets!
features_for_tweets = [(ngrams_features(t, frequency_dictionary, n), l) for (t, l) in tweets_and_labels]

# 4. Split data into training, test set
(train, test) = split_feature_set(features_for_tweets)

# 5. Train classifier!
classifier = nltk.NaiveBayesClassifier.train(train)
# classifier = nltk.DecisionTreeClassifier.train(train)

print classifier.show_most_informative_features(n=100)


input = 'i just got diabetes the other day'
input = input.lower()
input_list = tokenize_tweet(input)
print 'input is ' + str(input_list)
features = ngrams_features(input, frequency_dictionary, n)
print features
output = classifier.classify(features)
print 'output: ' + str(output)

prob = classifier.prob_classify(features)
print prob.prob(1)
print prob.prob(2)
print prob.prob(3)

train_accuracy = nltk.classify.accuracy(classifier, train)
test_accuracy = nltk.classify.accuracy(classifier, test)

print 'train accuracy:' + str(train_accuracy)
print 'test accuracy: ' + str(test_accuracy)

# fname1 = "../datatxt/unparsed_tweets_scraped.csv"
# f1 = open(fname1, 'r')
# content = f1.readlines()
# fname2 = "../datatxt/classified_tweets_1_2"
# f2 = open(fname2, 'w')
# for line in content:
# 	tweet = line.lower()
# 	tweet_tok = tokenize_tweet(tweet)
# 	feat = ngrams_features(line, frequency_dictionary, n)
# 	prediction = classifier.classify(feat)
# 	#prob = classifier.prob_classify(feat)
# 	#p = prob.prob(prediction)
# 	if prediction == 1 or prediction == 2:
# 		f2.write(line)
# 	#f2.write(str(prediction)+'\n')
# 	#f2.write(str(p)+'\n')

# f1.close()
# f2.close()
print "Classifying completed"
