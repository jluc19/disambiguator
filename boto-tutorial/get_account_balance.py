#!/bin/python
"""
Very simple example of using boto API to get your account balance.
This code is based on/shamelessly stone from the tutorial at
http://www.toforge.com/2011/04/boto-mturk-tutorial-create-hits/
"""

from boto.mturk.connection import MTurkConnection

#These keys are unique to your account, and can be found on Amazon Web Services under 'My Account' -> 'Security Credentials' 
ACCESS_ID = 'YOUR KEY'
SECRET_KEY = 'YOUR KEY'
HOST = 'mechanicalturk.sandbox.amazonaws.com'

#Set up a connection with MTurk 
conn = MTurkConnection(aws_access_key_id=ACCESS_ID, aws_secret_access_key=SECRET_KEY, host=HOST)
 
print 'Your current balance: ',  conn.get_account_balance()
