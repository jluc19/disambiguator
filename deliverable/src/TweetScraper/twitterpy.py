from TwitterSearch import *
import time
import csv

def scrape():
    try:
        tso = TwitterSearchOrder() # TwitterSearchOrder object
        tso.setKeywords(['diabetes']) # word queries (command line)
        tso.setLanguage('en') # English
        tso.setCount(100)
        tso.setIncludeEntities(False) # and don't give us all those entity information

        # it's about time to create a TwitterSearch object with our secret tokens
        ts = TwitterSearch(
            consumer_key = 'GJ4i8BuOTyvHw6HwLnF2kyvNI',
            consumer_secret = 'Zfks3yXnJ9f2rVCHaQJ9mdbjsQwECgEHihck5DC0aSyO0ibV9E',
            access_token = '286429860-Y9DQdFcnOhv9Dyzi16HTBs0h0E9g9ArcLTvWKZfr',
            access_token_secret = 'RT3f3iAP8Pdn5xkYWyOfOXzvrWvOwbdWDGMsY4F0yBXap'
         )
        counter = 0
        for tweet in ts.searchTweetsIterable(tso): # this is where the fun actually starts :)
            format_row(tweet['user']['screen_name'], tweet['text'])
            counter += 1
            if counter % 100 == 0:
                print counter, " tweets written"

    except TwitterSearchException as e: # take care of all those ugly errors if there are some
        print(e)

def format_row(user, tweet):
    print user
    print tweet
    t = time.strftime("%d_%m_%Y")
    filef = t + '_scrape.csv'
    with open(filef, 'a+') as output: 
        #wr = csv.writer(output, quoting=csv.QUOTE_NONE)
        #varRow = ['Tweet0','Tweet1','Tweet2','Tweet3','Tweet4','Tweet5','Tweet6','Tweet7','Tweet8','Tweet9','Tweet10','Tweet11','Tweet12','Tweet13','Tweet14','Tweet15','Tweet16','Tweet17','Tweet18','Tweet19','Tweet20']
        #wr.writerow(varRow)
        tweet = tweet.replace("'", '').strip()
        tweet = tweet.replace("\n", "")
        tweet = tweet.replace('"', '')
        joined = "@" + user + ", " + tweet
        tweet = ''.join(c for c in tweet if 0 < ord(c) < 127)
        row = ["@"+user, tweet]
        print row, "\n\n"
        wr = csv.writer(output, quoting=csv.QUOTE_ALL)
        wr.writerow(row)


scrape()