from itertools import chain

def hello():
	filename = 'cancer_tweets.csv'
	l = open('cancer_tweets.txt', 'w')
	
	with open(filename, 'r') as f:
		for line in f:
			if line is "":
				continue
			else:
				print line
				l.write(line)
		#line=[x.strip() for x in line if x]   # remove 'if x' if you want blank fields
    	#if len(line):
    	#	print line
        	#l.write(line)
	l.close()
	f.close()
hello()