import nltk
from nltk.corpus import stopwords

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

# Returns a list of all words in our corpus given a list of tuples of the 
# form ("This is a tweet", 3)
def get_word_list(corpus):
	words = []
	for (tweet, label) in corpus:
		words_in_tweet = tokenize_tweet(tweet)
		# print (words_in_tweet, label)
		words.append((words_in_tweet, label))
	return words

#Pull out all of the words in a list of tagged tweets, formatted in tuples.
def get_words_from_list(words_list):
	allwords = []
	for (words, sentiment) in words_list:
		allwords.extend(words)
	return allwords

#Order a list of tweets by their frequency.
def get_word_features(listoftweets):
	#Print out wordfreq if you want to have a look at the individual counts of words.
	wordfreq = nltk.FreqDist(listoftweets)
	words = wordfreq.keys()
	return words

def feature_extractor(doc):
	docwords = set(doc)
	features = {}
	for i in docwords:
		features['contains(%s)' % i] = True
	return features

filename = "../datatxt/parsed/labeled_tweets.txt"
data = parse_labeled_data(filename)

tweets = get_word_list(data)

(train, test) = split_feature_set(tweets)

allwords = get_words_from_list(tweets)
word_freqs = get_word_features(allwords)

wordlist = allwords

#Creates a training set - classifier learns distribution of true/falses in the input.
training_set = nltk.classify.apply_features(feature_extractor, train)

# classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)

print classifier.show_most_informative_features(n=100)

input = "got"
input = input.lower()
input = tokenize_tweet(input)
features = feature_extractor(input)
# print features
output = classifier.classify(features)
print input
print output
prob = classifier.prob_classify(features)
print prob.prob(1)
print prob.prob(2)
print prob.prob(3)

test_set = nltk.classify.apply_features(feature_extractor, test)

print 'test accuracy: ' + str(nltk.classify.accuracy(classifier, test_set))





