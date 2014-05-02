from boto.mturk.connection import MTurkConnection

ACCESS_ID = 'AKIAJDAK4I66WU7O6NVQ'
SECRET_KEY = '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'
HOST = 'mechanicalturk.sandbox.amazonaws.com'
 
def get_all_reviewable_hits(conn):
    hits = conn.get_reviewable_hits(page_size = 93)
    print "Total results to fetch %s " % hits.TotalNumResults
    return hits
   
 
conn = MTurkConnection(aws_access_key_id=ACCESS_ID, aws_secret_access_key=SECRET_KEY, host=HOST)
 
hits = get_all_reviewable_hits(conn)
#print hits
f = open('control_tweets_with_labels.txt')
file_content = f.readlines()
control_labels = {}

for line in file_content:
    splitted = line.split()
    label = splitted[len(splitted) - 1]
    control_labels[" ".join(line.split()[0:-1])] = label
    
f = open("rejected_tweets.txt", 'w')
f1 = open("accepted_tweets.txt", 'w')
for hit in hits:
    assignments = conn.get_assignments(hit.HITId)
    #conn.disable_hit(hit.HITId)
    approve = False
    for assignment in assignments:
        #print "Answers of the worker %s" % assignment.WorkerId
        for question_form_answer in assignment.answers[0]:
            question = question_form_answer.qid.replace('\n', '')
            answer = question_form_answer.fields[0]
           # print question
            if question in control_labels.keys():
              #  print 'here' 
                if answer == control_labels[question]:
                    approve = True

            

           # print '%s\t%s'%(question_form_answer.qid, question_form_answer.fields[0])
    
   

    if approve == False:
        for assignment in assignments:
            for question_form_answer in assignment.answers[0]:
                f.writelines(question_form_answer.qid.encode('ascii', 'ignore'))
            conn.reject_assignment(assignment.AssignmentId)
        conn.disable_hit(hit.HITId)
    else:
        for assignment in assignments:
            for question_form_answer in assignment.answers[0]:
                f1.write(question_form_answer.qid.encode('ascii', 'ignore'))
                f1.write(question_form_answer.fields[0].encode('ascii', 'ignore') + '\n')
            conn.approve_assignment(assignment.AssignmentId)
        conn.disable_hit(hit.HITId)
        

     

    	#Uncomment to approve assignment. Approving will remove this assignment from reviewable HITs, so store the data before approving 

    
    #Uncomment to remove all remaining assignments that have not been completed and approved/rejected
 
