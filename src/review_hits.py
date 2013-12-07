from boto.mturk.connection import MTurkConnection

ACCESS_ID = 'AKIAJDAK4I66WU7O6NVQ'
SECRET_KEY = ''
HOST = 'mechanicalturk.sandbox.amazonaws.com'
 
def get_all_reviewable_hits(conn):
    page_size = 50
    hits = conn.get_reviewable_hits()
    print "Total results to fetch %s " % hits.TotalNumResults
    return hits
   
 
conn = MTurkConnection(aws_access_key_id=ACCESS_ID, aws_secret_access_key=SECRET_KEY, host=HOST)
 
hits = get_all_reviewable_hits(conn)

for hit in hits:
    assignments = conn.get_assignments(hit.HITId)
    for assignment in assignments:
        approve = False
        print "Answers of the worker %s" % assignment.WorkerId
        for question_form_answer in assignment.answers[0]:
            if question_form_answer.qid == 'quality control' and question_form_answer[0] == 3:
                approve = True

            print '%s\t%s'%(question_form_answer.qid, question_form_answer.fields[0])


    	#Uncomment to approve assignment. Approving will remove this assignment from reviewable HITs, so store the data before approving 

	#conn.approve_assignment(assignment.AssignmentId)
    
    #Uncomment to remove all remaining assignments that have not been completed and approved/rejected
    #conn.disable_hit(hit.HITId)
 
