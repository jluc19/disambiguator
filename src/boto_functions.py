import langid
from boto.mturk.connection import MTurkConnection
from boto.mturk.question import QuestionContent,Question,QuestionForm,Overview,AnswerSpecification,SelectionAnswer,FormattedContent,FreeTextAnswer
 
ACCESS_ID ='AKIAJDAK4I66WU7O6NVQ'
SECRET_KEY = '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'
HOST = 'mechanicalturk.sandbox.amazonaws.com'
 
mtc = MTurkConnection(aws_access_key_id=ACCESS_ID,
                      aws_secret_access_key=SECRET_KEY,
                      host=HOST)

def format_tweets():
	f = open('unparsed_tweets_scraped.csv', 'r')
	file_content = f.readlines()
	
	
	w = open('parsed_tweets_scraped.csv', 'w')

	non_dup = list(set(file_content))
	for line in non_dup:
		line = line.replace("\n", "")
		tup = langid.classify(line)
		if "en" in tup:
			line = line.translate(None, '"')
			line = '\"' + line + '\"'
			if "diabetes" in line:
				if not line.startswith("\"RT"):
					if not "http" in line and not "https" in line:
						w.write(line + "\n")

	f.close()
	w.close()

def delete_all_hits():
    hits = mtc.get_all_hits()
    print hits
    for hit in hits:
    	mtc.disable_hit(hit.HITId)


format_tweets()