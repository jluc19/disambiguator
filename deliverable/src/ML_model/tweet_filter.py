import sys
import svm_tweet_classifier_object

# IO
# 
# opens file with usernames, tweets
# file SHOULD HAVE line1: user, line2: tweet, line3: user, ...
# returns a list of (username, tweet)
def IO():
	if len(sys.argv) < 2:
		print "Usage: tweet_filter.py <file to with username, tweets>"
		return []
	else:
		f = sys.argv[1]
		inp = open(f, 'r')

		samples = []

		ln   = 1
		line = inp.readline()
		while line:
			line = line.rstrip('\n')

			if (ln % 2 == 1):
				username = line
			else:
				tweet = line
				elem = (username, tweet)
				samples.append(elem)
				print 'user: ' + username + ', tweet: ' + tweet
			line = inp.readline()
			ln = ln + 1
	    
		return samples

def filter_tweets(input_data, ml_model):
	tweets = [t for (u, t) in input_data]
	users  = [u for (u, t) in input_data]

	features = svm_tweet_classifier_object.get_features(tweets)
	labels = ml_model.predict(t)
	user_labels = {}

	for i in range(len(labels))
		user = users[i]
		label = labels[i]

		if user in user_labels:
			l = user_labels[user]
			l.append(label)
			user_labels[user] = l
		else:
			user_labels[user] = [label]
	#for each tweet/user pair, feed the tweet into the ML machine, return the tag, and add the tag to the user's array
	#TODO: check to make sure at least one of the keywords exists in the tweet before you send it to ML algo
	return user_labels

def analyze_data(results):
	self = []
	other = []

	for user in results:
		labels = results[user]
		for label in labels:
			if label == 1 or label == 2:
				concern.append(user)
	#if some % of relevant tweets to that user fit into a category, add them to that users category
	#print output of three arrays to console and save data to a CSV (username & relevant tweets)
	#discard all tweets that do not fit the % relevance for any category
	#discuss assumptions with Lyle and Andy STAT
	return [self, other]

samples = IO()
ml_model = svm_tweet_classifier_object.run()
ML_results = filter_tweets(samples, ml_model)
users = analyze_data(ML_results)
selfs = users[0]
others= users[1]
print "Analysis complete. Total users found: " + str(len(selfs) + len(others))
print "Potentials with disease: " + str(len(selfs)) + "\n"
for user in selfs:
	print user + "\n"
print "Potentials knowing person with disease: " + str(len(others)) + "\n"
for user in others:
	print user + "\n"
