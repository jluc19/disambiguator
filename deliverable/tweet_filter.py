import sys

def IO():
	if len(sys.argv) < 2:
		print "Usage: tweet_filter.py <file to be analyzed>"
	else:
		f = sys.argv[1]
		inp = open(f, 'r')
		while(1):
			lines = inp.readlines(1000)
			#get tweet and user as a pairwise value
			#check that the tweet involves the disease (have the user input the disease or keywords they want to use to filter)
			#perhaps create a dictionary of tweets to usernames? Need to handle duplicates

	return 0
def filter_tweets(input_data):
	#for each tweet/user pair, feed the tweet into the ML machine, return the tag, and add the tag to the user's array
	#check to make sure at least one of the keywords exists in the tweet before you send it to ML algo
	print "text"
	return 0

def analyze_data(results):
	#if some % of relevant tweets to that user fit into a category, add them to that users category
	#print output of three arrays to console and save data to a CSV (username & relevant tweets)
	#discard all tweets that do not fit the % relevance for any category
	#discuss assumptions with Lyle and Andy STAT
	return [1,2,8]

input_data = IO()
ML_results = filter_tweets(input_data)
sizes = analyze_data(ML_results)
print "Analysis complete. Total users found:"
print "Potentials with disease: " + sizes[0] + "\n"
print "Potentials knowing person with disease: " + sizes[1]