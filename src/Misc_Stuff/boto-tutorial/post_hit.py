#!/bin/python
"""
Very simple example of using boto API to create and post a HIT
This code is based on/shamelessly stone from the tutorial at
http://www.toforge.com/2011/04/boto-mturk-tutorial-create-hits/
"""

from boto.mturk.connection import MTurkConnection
from boto.mturk.question import QuestionContent,Question,QuestionForm,Overview,AnswerSpecification,SelectionAnswer,FormattedContent,FreeTextAnswer
 
#These keys are unique to your account, and can be found on Amazon Web Services under 'My Account' -> 'Security Credentials' 
ACCESS_ID = 'YOUR KEY'
SECRET_KEY = 'YOUR KEY'
HOST = 'mechanicalturk.sandbox.amazonaws.com'
 
#Set up a connection with MTurk 
conn = MTurkConnection(aws_access_key_id=ACCESS_ID, aws_secret_access_key=SECRET_KEY, host=HOST)

title = "Grade my students' assignments."
description = ('I am a very lazy TA. Please do my work for me.')

#These are the keywords most associated with high volume on MTurk, according to last week's lecture...
keywords = 'data, collection, easy, writing, transcribe'
 

#Overview, to be displayed at the top of the HIT
 
overview = Overview()
overview.append_field('Title', 'Give your opinion on this website')
  
#Create the question

students = ['Shadia', 'Christina', 'Matthew', 'Quanze', 'Casey', 'Manosai', 'Lewis', 'Tiernan', 'Joel', 'Susan', 'Alex', 'Evan', 'Daniel', 'Chenyang', 'Corey', 'Jason', 'Tommy', 'Varshil', 'Crystal', 'Sunny', 'Jiten', 'Taylor', 'Neil']

grades =[('A','A'), ('B','B'), ('C','C'), ('D','D'), ('FAIL!','F')]

questions = []

#add one question for each student
for student in students : 
	qc = QuestionContent()
	qc.append_field('Title','What grade does %s deserve?'%student)
 
	a = SelectionAnswer(min=1, max=1,style='dropdown', selections=grades, type='text', other=False) 
	q = Question(identifier=student, content=qc, answer_spec=AnswerSpecification(a), is_required=True)
	questions.append(q)
 
#Create the question form

question_form = QuestionForm()
question_form.append(overview)
for q in questions: question_form.append(q)
 
#post the HIT
 
conn.create_hit(questions=question_form, max_assignments=1, title=title, description=description, keywords=keywords, duration = 60*5, reward=0.05)
