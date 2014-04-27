# -*- coding: utf-8 -*-
import langid, random, csv, re
from boto.mturk.connection import MTurkConnection
from boto.mturk.question import QuestionContent,Question,QuestionForm,Overview,AnswerSpecification,SelectionAnswer,FormattedContent,FreeTextAnswer
 
ACCESS_ID ='AKIAJDAK4I66WU7O6NVQ'
SECRET_KEY = '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'
HOST = 'mechanicalturk.sandbox.amazonaws.com'
 
mtc = MTurkConnection(aws_access_key_id=ACCESS_ID,
                      aws_secret_access_key=SECRET_KEY,
                      host=HOST)

def process_file_content(file_content):
	ret = []
	for line in file_content:
		line = re.sub('@[^\s]+','USER',line)
		line = re.sub('[\s]+', ' ', line)
		line = re.sub(r'#([^\s]+)', r'\1', line)
		ret.append(line)
	return ret

def format_tweets():
	f = open('../datatxt/raw/all_tweets.txt', 'r')
	w = open('parsed_tweets_scraped.csv', 'w')
	l = open('parsed_tweets_scraped_links.csv', 'w')
	while 1:
		file_content = f.readlines(10000)
		if not file_content:
			break
		file_content = process_file_content(file_content)
		non_dup = list(set(file_content))
		for line in non_dup:
			tup = langid.classify(line)
			#print tup
			if "en" in tup:
				if "diabetes" in line:
					#print line
					line = line.replace("'", '').strip()
					line = line.replace('"', '')
					#print line
					if not line.startswith("\"RT"):
						if not "http" in line and not "https" in line:
							w.write(line + "\n")
						else:
							line = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''','',line)
							l.write(line + "\n")
	f.close()
	w.close()
	l.close()
	w = open('parsed_tweets_scraped_links.csv', 'r')
	print w.readlines()
	w.close()

def delete_all_hits():
    hits = mtc.get_all_hits()
    print hits
    for hit in hits:
    	mtc.disable_hit(hit.HITId)

def format_csv():
	f = open('parsed_tweets_scraped.csv', 'r')
	c = open('control_tweets.csv', 'r')
	control = c.readlines()
	output = open('tweets_for_mTurk.csv', 'wb')
	wr = csv.writer(output, quoting=csv.QUOTE_NONE)
	varRow = ['Tweet0','Tweet1','Tweet2','Tweet3','Tweet4','Tweet5','Tweet6','Tweet7','Tweet8','Tweet9','Tweet10','Tweet11','Tweet12','Tweet13','Tweet14','Tweet15','Tweet16','Tweet17','Tweet18','Tweet19','Tweet20']
	wr.writerow(varRow)
	wr = csv.writer(output, quoting=csv.QUOTE_ALL)
	while 1:
		hit = []
		for i in range(20):
			if(i==5 or i == 15):
				hit.append(random.choice(control).strip())
			else:
				line = f.readline().strip()
				if not line: return 0
				line = line.replace("'", '').strip()
				line = line.replace('"', '')
				hit.append(line)
		wr.writerow(hit)


format_tweets()
format_csv()