from boto.mturk.connection import MTurkConnection
from boto.mturk.question import QuestionContent,Question,QuestionForm,Overview,AnswerSpecification,SelectionAnswer,FormattedContent,FreeTextAnswer
import random

ACCESS_ID ='AKIAJDAK4I66WU7O6NVQ'
SECRET_KEY = ''
HOST = 'mechanicalturk.sandbox.amazonaws.com'
 
mtc = MTurkConnection(aws_access_key_id=ACCESS_ID,
                      aws_secret_access_key=SECRET_KEY,
                      host=HOST)
 
title = 'Diabetes Tweet Annotation'
description = ('Read a list of tweets and label each tweet'
               'based on the content of the tweet')
keywords = 'diabetes, twitter, tweets, annotation'
 
ratings =[('Sick', '1'),
          ('General', '2'),
         ('Unrelated', '3')]
 
#---------------  BUILD OVERVIEW -------------------
 
overview = Overview()
overview.append_field('Title', 'Disease Tweet Annotation')
overview.append_field('Text', 'For each tweet, please specify the following:\n'
                                      'If the tweeter has diabetes label Sick.\n'
                                      'If it\'s a general statement about diabetes label General.\n'
                                      'If it\'s unrelated, label Unrelated.\n')


#--------------- BUILD QUESTIONS ----------------------- 


f = open('diabetes_parsed.txt', 'r')
file_content = f.readlines()

question_form = QuestionForm()
question_form.append(overview)


i = 0
rand = random.randint(1,20)
for tweet in file_content:
  
  if i == 20:
    rand = random.randint(1,20)
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
    qual_qc.append_field('Text', "Please label this Unrelated")

    qual = Question(identifier='quality control',
                  content=qual_qc,
                  answer_spec=AnswerSpecification(fta),
                  is_required=True)
    question_form.append(qual)
#q1.template = '<Question><Length minLength="2" maxLength="2" /></Question>'
  i+=1
 
