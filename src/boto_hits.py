from boto.mturk.connection import MTurkConnection
from boto.mturk.question import QuestionContent,Question,QuestionForm,Overview,AnswerSpecification,SelectionAnswer,FormattedContent,FreeTextAnswer
import random

ACCESS_ID ='AKIAJDAK4I66WU7O6NVQ'
SECRET_KEY = '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'
HOST = 'mechanicalturk.sandbox.amazonaws.com'
 
mtc = MTurkConnection(aws_access_key_id=ACCESS_ID,
                      aws_secret_access_key=SECRET_KEY,
                      host=HOST)
 
title = 'Diabetes Tweet Annotation'
description = ('Read a list of tweets and label each tweet'
               'based on the content of the tweet')
keywords = 'diabetes, twitter, tweets, annotation'
 
ratings =[('Self', '1'),
          ('Other Person', '2'),
          ('General', '3'),
         ('Unrelated', '4')]
 
#---------------  BUILD OVERVIEW -------------------
 
overview = Overview()
overview.append_field('Title', 'Disease Tweet Annotation')
overview.append_field('Text', 'For each tweet, please specify the following:\n'
                                      'Only if the tweeter has diabetes, label Self.\n'
                                      'If the tweeter is referring to another individual with diabetes, label Other Person\n'
                                      'If it\'s a general statement about diabetes label, General.\n'
                                      'If it\'s unrelated or not in English, label Unrelated.\n'
                                      'For example, \'I don\'t know if I threw up because of dairy or diabetes. Help. Me.\' should be labeled as Self')


#--------------- BUILD QUESTIONS ----------------------- 


f = open('diabetes_parsed.txt', 'r')
file_content = f.readlines()
f.close()

f_2 = open('control_tweets.txt', 'r')
control_content = f_2.readlines()
f_2.close()

question_form = QuestionForm()
question_form.append(overview)


i = 0
rand = random.randint(1,20)
control = control_content[random.randint(0,2)]
for tweet in file_content:
  
  if i == 20:
    rand = random.randint(1,20)
    rand_control = random.randint(0,2)
    control = control_content[rand_control]
    i = 0
 
    #--------------- CREATE THE HIT -------------------
     
    mtc.create_hit(questions=question_form,
                   max_assignments=1,
                   title=title,
                   description=description,
                   keywords=keywords,
                   duration = 60*5,
                   reward=0.02)
 
    question_form = QuestionForm()
    question_form.append(overview)

  qc = QuestionContent()
  qc.append_field('Text', tweet)
     
  fta = SelectionAnswer(min=1, max=1,style='radiobutton',
                          selections=ratings,
                          type='text',
                          other=False)
     
  q = Question(identifier=tweet,
                  content=qc,
                  answer_spec=AnswerSpecification(fta),
                  is_required=True)
  question_form.append(q)

  if i == rand:
    qual_qc = QuestionContent()
    qual_qc.append_field('Text', control)

    qual = Question(identifier=control,
                  content=qual_qc,
                  answer_spec=AnswerSpecification(fta),
                  is_required=True)
    question_form.append(qual)
#q1.template = '<Question><Length minLength="2" maxLength="2" /></Question>'
  i+=1
 