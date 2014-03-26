#Tweet classifier using SVM
#Boyang Zhang and Jason Lucibello

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

corpus = ['I have diabetes',
			'You have diabetes',
			'He has diabetes',
			'She has diabetes']

vectorizer = CountVectorizer(min_df=1, ngram_range=(1,2))

X = vectorizer.fit_transform(corpus)
analyze = vectorizer.build_analyzer()
print analyze("I have diabetes")
print X
