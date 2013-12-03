from boto.mturk.connection import MTurkConnection
from boto.mturk.question import QuestionContent,Question,QuestionForm,Overview,AnswerSpecification,SelectionAnswer,FormattedContent,FreeTextAnswer
 
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
          ('General', '1'),
         ('Unrelated', '1')]
 
#---------------  BUILD OVERVIEW -------------------
 
overview = Overview()
overview.append_field('Title', 'Disease Tweet Annotation')
overview.append_field('Text', 'For each tweet, please specify the following:\n'
                                      'If the tweeter has diabetes label Sick.\n'
                                      'If it\'s a general statement about diabetes label General.\n'
                                      'If it\'s unrelated, label Unrelated.\n')


#--------------- BUILD QUESTIONS ----------------------- 
 #What are all the fields?
qc1 = QuestionContent()
qc1.append_field('Text','@funnyorfact: If Jim has 50 chocolate bars, and eats 45, what does he have? Diabetes. Jim has diabetes..')
 
fta1 = SelectionAnswer(min=1, max=1,style='radiobutton',
                      selections=ratings,
                      type='text',
                      other=False)
 
q1 = Question(identifier='identifier',
              content=qc1,
              answer_spec=AnswerSpecification(fta1),
              is_required=True)
#q1.template = '<Question><Length minLength="2" maxLength="2" /></Question>'
 
 
#--------------- BUILD THE QUESTION FORM -------------------
 
question_form = QuestionForm()
question_form.append(overview)
question_form.append(q1)
 
#--------------- CREATE THE HIT -------------------
 
mtc.create_hit(questions=question_form,
               max_assignments=1,
               title=title,
               description=description,
               keywords=keywords,
               duration = 60*5,
               reward=0.02)