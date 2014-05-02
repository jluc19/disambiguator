#Disease Classifier V1.0 4/30/2014
Identifying Clinical Trial Candidates through Machine Learning

Jason Lucibello, Nathan Fraenkel, Varun Gupta, Dan Zhang


# Disease Classifier
The classifier is divided into 4 modules.
(1) Twitter scraper
(2) Mechanical Turk Package
(3) Annotated training data
(4) Machine Learning Model.

##Twitter Scraper
Our twitter scraper uses the TwitterSearch python library (https://github.com/ckoepp/TwitterSearch) to access Twitter.com's firehose and search real-time tweets by language, location, and keyword.

##Mechanical Turk Package
Packages tweets gathered from the twitter scraper for annotation by Mechanical Turk. MTurk only accepts files in UTF-8 format, which may require using Save As... depending on your text editor.

##Machine Learning Model
Our model uses a Linear Support Vector Machine to classify tweets into 3 categories.

Self: the tweeter has the chosen disease
Other: the tweeter refers to another person with the chosen disease
General: the tweeter speaks about the disease in general

Our features are composed of tweet unigrams and bigrams.

Feature selection uses a combination of Lemmatization, stemming, Chi-squared and univariate feature selection, reducing the number of features from ~27,000 to ~350.

##Annotated Training Data
Data output from Mechanical Turk, manual annotation, is placed in \training_data\labeled_data. Currently our training set is diabetes_training_set.txt, but similar training sets can be created for cancer, heart disease, etc.

Training data should be of the form

	Tweet
	Annotation

# Usage
1. Python Scraper: scrapes twitter.com for tweets based on keywords

In \twitter_scraper, 

 	python twitter_scraper.py
	
Enter chosen disease

2. MTurk Package

3. ML Model: Takes in training and test set as input and classifies tweets into 3 categories. Output is text files of tweet/user pairs

Identify annotated training dataset and set in line 24 of \ML_model\ml_classifier.py

Identify unknown test dataset from Twitter Scraper and set in line 239 of ML_model\ml_classifier.py

In \ML_model,

	python ml_classifier.py

Ouputs for each class of the model will be found in \ML_model\classifications

# Installation
The tool requires the following Python Libraries:

1. TwitterSearch

	pip install TwitterSearch

2. Languid: https://github.com/saffsd/langid.py

3. Scikit-learn: http://scikit-learn.org/stable/install.html

Note: On Mac OS X 10.9, flags must be added for the compilation to work.

	# Pre-requsites:
	# X-code 5.0.1 with command line tools
	# Homebrew, homebrew python, homebrew pip
	 
	export CFLAGS="-arch i386 -arch x86_64"
	export FFLAGS="-m32 -m64"
	export LDFLAGS="-Wall -undefined dynamic_lookup -bundle -arch i386 -arch x86_64"
	export CC=gcc-4.2
	export CXX="g++ -arch i386 -arch x86_64"
	brew install gfortran
	brew install apple-gcc42
	pip install cython
	 
	#use existing brews
	brew tap samueljohn/python
	brew tap homebrew/science
	pip install nose
	 
	#installs numpy as a dependency
	brew install scipy
	 
	pip install scikit-learn

(From https://gist.github.com/xlfe/7786057)

4. MatPlotLib: http://matplotlib.org/downloads.html

This is used to view graphics output of Scikit-learn.

5.NLTK

	sudo pip install -U numpy
	sudo pip install -U pyyaml nltk

## Contributors
Created by Jason Lucibello, Nathan Fraenkel, Varun Gupta, and Dan Zhang at the University of Pennsylvania 2014
