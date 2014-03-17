import csv

def review_hits():
	f_accept = open("results_accepted.csv", 'w')
	f_reject = open("results_rejected.csv", 'w')

	start = True
	for line_list in csv.reader(open("Batch_1456545_batch_results.csv", "r")):
		if start:
			start = False
		else:
			tweet_label_dict = {}
			control1 = False
			control2 = False
			tweets = line_list[27:47]
			q1_list = line_list[-100:-80]
			q2_list = line_list[-80:-60]
			q3_list = line_list[-60:-40]
			q4_list = line_list[-40:-20]
			q5_list = line_list[-20:]
			
			for i in range(0,20):
				label = 0

				if q1_list[i] is "No" or q4_list[i] == "Yes":
					if q5_list[i] == "Yes":
						label = '4'
					else:
						label = '3'
				elif q1_list[i] == "Yes" and q3_list[i] == "Yes": 
					if q2_list[i] == "Themselves":
						label = '1'
					else:
						label = '2'
				else:
					if q5_list[i] == '"Yes"':
						label = '4'
					else:
						label = '3'


				if i == 6:
					control1 = compare_control(tweets[i], label)
				elif i == 14:
					control2 = compare_control(tweets[i], label)
				else:
					tweet_label_dict[tweets[i]] = label


				#if both controls match, then accept else reject	
			if control1 and control2:
				for key, value in tweet_label_dict.iteritems():
					f_accept.write(key + '\n')
					f_accept.write(value + '\n')
			else:
				for key, value in tweet_label_dict.iteritems():
					f_reject.write(key + '\n')
					f_reject.write(value + '\n')
			
	f_accept.close()
	f_reject.close()


def compare_control(tweet, label):
	if label is '4':
		label = '3'

	# issue with utf encoding characters for one tweet
	if "Yall still making excuses" in tweet:
		value = (v for (k,v) in control_dict.iteritems() if 'Yall still making excuses' in k).next()
		
		if label == value:
			return True
		else:
			print label + "\t" + tweet + "\t" + value
			return False
	#all other control tweets
	else:
		if 'I think Ive got diabetes' in tweet or 'Diagnosed myself with diabetes' in tweet or 'I think Ive got diabetes' in tweet:
			print "here"
			if label == '3':
				return True
		value = control_dict[tweet]
		if label == value:
			return True
		else:
			print label + "\t" + tweet + "\t" + control_dict[tweet]
			return False
	

f_control = open("control_tweets_with_labels.txt", 'r')
control_dict = {}
for line in f_control:
	control_dict[line.split('\t')[0]] = line.split('\t')[1].replace("\n", "")

f_control.close()
review_hits()









#1: tweet about person?
#2: themselves or another person
#3: Diabetes?
#4: Joke?
#5: Diabetes as disease

#if 1 is no or 4 is yes, then 3
#if 3 and 5 is yes, then 4
#if 1 is yes, 2 is themselves, then 1
#if 1 is yes, 2 is other person, then 2

