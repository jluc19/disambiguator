from TwitterSearch import *
import time
import csv

def scrape():
    try:
        tso = TwitterSearchOrder() # TwitterSearchOrder object
        keyword = raw_input('Enter disease to query: ')
        tso.setKeywords([keyword]) # word queries (command line)
        tso.setLanguage('en') # English
        tso.setIncludeEntities(False) # ignore entity information
        tso.setCount(100) #page count

        ts = TwitterSearch(
            consumer_key = 'GJ4i8BuOTyvHw6HwLnF2kyvNI',
            consumer_secret = 'Zfks3yXnJ9f2rVCHaQJ9mdbjsQwECgEHihck5DC0aSyO0ibV9E',
            access_token = '286429860-Y9DQdFcnOhv9Dyzi16HTBs0h0E9g9ArcLTvWKZfr',
            access_token_secret = 'RT3f3iAP8Pdn5xkYWyOfOXzvrWvOwbdWDGMsY4F0yBXap'
         )
        counter = 0
        for tweet in ts.searchTweetsIterable(tso):
            format_row(tweet['user']['screen_name'], tweet['text'])
            counter += 1
            if counter % 100 == 0:
                print counter, " tweets written"
    except TwitterSearchException as e:
        print(e)

def format_row(user, tweet):
    t = time.strftime("%d_%m_%Y")
    filef = t + '_scrape.csv'
    with open(filef, 'a+') as output: 
        tweet = tweet.replace("'", '').strip()
        tweet = tweet.replace("\n", "")
        tweet = tweet.replace('"', '')
        joined = "@" + user + ", " + tweet
        tweet = ''.join(c for c in tweet if 0 < ord(c) < 127)
        row = ["@"+user, tweet]
        wr = csv.writer(output, quoting=csv.QUOTE_ALL)
        wr.writerow(row)
scrape()