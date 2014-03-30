# -*- coding: utf-8 -*-
import langid, random, csv, re

def reformat_tweets():
	w = open('labeled_tweets.txt', 'r')
	l = open('new_labeled_tweets.txt', 'w')
	while 1:
		file_content = w.readlines(10000)
		if not file_content:
			break
		i = 0
		for line in file_content:
			if line.startswith("###"):
				continue
			print i, line
			if i % 2 == 1:
				line = re.sub('@[^\s]+','USER',line)
				line = re.sub('[\s]+', ' ', line)
				line = re.sub(r'#([^\s]+)', r'\1', line)
				line = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''','',line)
			else:
				l.write("\n")
			i = i + 1
			print line
			l.write(line)
	w.close()
	l.close()

reformat_tweets()