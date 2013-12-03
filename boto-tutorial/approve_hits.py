from boto.mturk.connection import MTurkConnection

ACCESS_ID = 'YOUR KEY'
SECRET_KEY = 'YOUR KEY'
HOST = 'mechanicalturk.sandbox.amazonaws.com'
 
def get_all_reviewable_hits(conn):
    page_size = 50
    hits = conn.get_reviewable_hits(page_size=page_size)
    print "Total results to fetch %s " % hits.TotalNumResults
    print "Request hits page %i" % 1
    total_pages = float(hits.TotalNumResults)/page_size
    int_total= int(total_pages)
    if(total_pages-int_total>0):
        total_pages = int_total+1
    else:
        total_pages = int_total
    pn = 1
    while pn < total_pages:
        pn = pn + 1
        print "Request hits page %i" % pn
        temp_hits = conn.get_reviewable_hits(page_size=page_size,page_number=pn)
        hits.extend(temp_hits)
    return hits
 
conn = MTurkConnection(aws_access_key_id=ACCESS_ID, aws_secret_access_key=SECRET_KEY, host=HOST)
 
hits = get_all_reviewable_hits(conn)

for hit in hits:
    assignments = conn.get_assignments(hit.HITId)
    for assignment in assignments:
        print "Answers of the worker %s" % assignment.WorkerId
        for question_form_answer in assignment.answers[0]:
		print '%s\t%s'%(question_form_answer.qid, question_form_answer.fields[0])

    	#Uncomment to approve assignment. Approving will remove this assignment from reviewable HITs, so store the data before approving 
	conn.approve_assignment(assignment.AssignmentId)
    
    #Uncomment to remove all remaining assignments that have not been completed and approved/rejected
    conn.disable_hit(hit.HITId)
 
