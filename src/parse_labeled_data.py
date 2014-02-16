import nltk

def parse_labeled_data(filename):

	f = open(filename, 'r')
	file_content = f.readlines()

	i = 0
	tweet_and_label_map = {}
	tweet = ''
	label = ''
	for line in file_content:
		line = line.rstrip('\n')
		if i % 2 == 0:
			tweet = line
		else:
			label = line
			tweet_and_label_map[tweet] = label
		i = i + 1

	return tweet_and_label_map

def tokenize_tweet(tweet):
	print tweet
	tokens = nltk.word_tokenize(tweet)
	# tagged = nltk.pos_tag(tokens)
	return tokens

filename = "../datatxt/parsed/nate.txt"
dictionary = parse_labeled_data(filename)

tweet_number = 1
tweet = dictionary.keys()[tweet_number]
print tweet

tokens = tokenize_tweet(tweet)
print tokens