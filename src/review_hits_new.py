import csv
import sys

def review_hits(filename):
	f_accept = open("results_accepted.csv", 'w')
	f_reject = open("results_rejected1.csv", 'w')

	start = True
	f_approve = open("approval_"+filename, "w")
	newline = ""

	for line_list in csv.reader(open(filename, "r")):
		if start:
			print line_list[48:68]
			print line_list[68:88]
			print line_list[88:108]
			print line_list[108:128]
			print line_list[128:148]
			newline = ','.join(line_list)
			f_approve.write(newline)
			start = False
		else:
			tweet_label_dict = {}
			control1 = False
			control2 = False
			tweets = line_list[27:48]
			q1_list = line_list[48:68]
			q2_list = line_list[68:88]
			q3_list = line_list[88:108]
			q4_list = line_list[108:128]
			q5_list = line_list[128:148]
			
			for i in range(0,20):
				label = 0

				if q1_list[i] == "Yes" and q3_list[i] == "Yes": 
					if q2_list[i] == "Themselves":
						label = '1'
					else:
						label = '2'
				elif q1_list[i] is "No" or q4_list[i] == "Yes":
					if q5_list[i] == "Yes":
						label = '4'
					else:
						label = '3'
				
				else:
					if q5_list[i] == '"Yes"':
						label = '4'
					else:
						label = '3'

				
				if i == 5:
					print tweets[i]
					print q1_list[i]
					print q2_list[i]
					print q3_list[i]
					print q4_list[i]
					print q5_list[i]
					print label
					control1 = compare_control(tweets[i], label)
				elif i == 15:
					print tweets[i]
					print q1_list[i]
					print q2_list[i]
					print q3_list[i]
					print q4_list[i]
					print q5_list[i]
					print label
					control2 = compare_control(tweets[i], label)
				else:
					tweet_label_dict[tweets[i]] = label


				#if both controls match, then accept else reject	
			if control1 and control2:
				for key, value in tweet_label_dict.iteritems():
					f_accept.write(key + '\n')
					f_accept.write(value + '\n')
				newline = ','.join(line_list)
				newline = newline + ',X, '
			else:
				for key, value in tweet_label_dict.iteritems():
					f_reject.write(key + '\n')
					f_reject.write(value + '\n')
				newline = ','.join(line_list)
				newline = newline + ',,HIT was not completed accurately'
		f_approve.write(newline + '\n')	
	f_accept.close()
	f_reject.close()
	f_approve.close()


def compare_control(tweet, label):
	if label is '4':
		label = '3'

	print "label"
	print label
	# issue with utf encoding characters for one tweet
	if "Yall still making excuses" in tweet:
		value = (v for (k,v) in control_dict.iteritems() if 'Yall still making excuses' in k).next()

		if label == value:
			print "------------------------------------------------------"

			return True
		else:
			print "value"
			print value
			#print label + "\t" + tweet + "\t" + value
			print "------------------------------------------------------"

			return False
	#all other control tweets
	else:
		if 'I think Ive got diabetes' in tweet or 'Diagnosed myself with diabetes' in tweet or 'I think Ive got diabetes' in tweet:
			if label == '3':
				print "------------------------------------------------------"

				return True
		value = control_dict[tweet]
		
		if label == value:
			print "------------------------------------------------------"

			return True
		else:
			print "value"
			print value
			#print label + "\t" + tweet + "\t" + control_dict[tweet]
			print "------------------------------------------------------"

			return False

f_control = open("control_tweets_with_labels.txt", 'r')
control_dict = {}
for line in f_control:
	key = line.split('\t')[0]
	control_dict[key] = line.split('\t')[1].replace('\n', '')

f_control.close()
review_hits(sys.argv[1])









#1: tweet about person?
#2: themselves or another person
#3: Diabetes?
#4: Joke?
#5: Diabetes as disease

#if 1 is no or 4 is yes, then 3
#if 3 and 5 is yes, then 4
#if 1 is yes, 2 is themselves, then 1
#if 1 is yes, 2 is other person, then 2

