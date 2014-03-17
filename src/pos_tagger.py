import nltk
import CMUTweetTagger

ex1 = "Before what happened at lunch when someone decided to piss me off.I had so much at the diabetes walk. Especially with my girls #TWERKTEAM!"
#text1 = nltk.word_tokenize(ex1)
#print nltk.pos_tag(text1)

ex2 = "my mum got me a tofee apple company.. Diabetes is definately tryna proceed to me LOL"
#text2 = nltk.word_tokenize(ex2)
#print nltk.pos_tag(text2)

print CMUTweetTagger.runtagger_parse(["my mum got me a tofee apple company.. Diabetes is definately tryna proceed to me LOL"])