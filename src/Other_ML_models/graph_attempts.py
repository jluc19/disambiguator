# -*- coding: utf-8 -*-
#Tweet classifier using SVM
#Boyang Zhang and Jason Lucibello

import nltk
import numpy as np
#this might be a challenge to install
#from mayavi import mlab
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn import svm, grid_search
from sklearn.datasets import make_moons
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFwe
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import random, re, collections, itertools
from matplotlib import cm, mlab
from sklearn import svm, datasets

from matplotlib.ticker import LinearLocator, FormatStrFormatter

from scipy.sparse import coo_matrix

sentiments = [1 ,2, 3]
target_names = ["Self", "Another Person", "General Statement"]
dv = DictVectorizer()
le = LabelEncoder()

def removeNonAscii(s): return "".join(i for i in s if ord(i)<128)

def plot_coo_matrix(m):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    return ax


def parse_labeled_data(filename):
	#variable setup
	ones, twos, threes, tweets_and_labels = ([] for i in range(4))
	tweet, label = '', ''
	i = 1
	newFile = open('../training_data/ordered_tweets.txt', 'w')
	dup = open('duplicates.txt', 'w')
	with open(filename, 'r') as f:
		for line in f:
			if line.startswith('###'):
				continue
			line = line.rstrip('\n')
			#removeNonAscii(line)
			#print line
			if i % 2 == 1:
				line = re.sub('@[^\s]+','USER',line)
				line = re.sub("^\s+","", line)
				line = re.sub(r'#([^\s]+)', r'\1', line)
				#line = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''','',line)
				tweet = line
			else:
				l = int(line)
				if l == 1:
					ones.append((tweet, l))
				elif l == 2:
					twos.append((tweet, l))
				else:
					threes.append((tweet, 3))
			i = i + 1


	duplicates = []
	duplicates.extend(ones)
	duplicates.extend(twos)
	duplicates.extend(threes)

	dup.write(str([x for x, y in collections.Counter(duplicates).items() if y > 1]))

	#remove duplicates
	ones = list(set(ones))
	twos = list(set(twos))
	threes = list(set(threes))

	for item, val in ones:
		newFile.write(item + "\n")
		newFile.write(str(val) + "\n")
	for item, val in twos:
		newFile.write(item + "\n")
		newFile.write(str(val) + "\n")
	for item, val in threes:
		newFile.write(item + "\n")
		newFile.write(str(val) + "\n")
	newFile.close()

	smallest = min([len(l) for l in [ones, twos, threes]])
	print 'we have ' + str(len(ones)) + ' tweets labeled with a 1'
	print 'we have ' + str(len(twos)) + ' tweets labeled with a 2'
	print 'we have ' + str(len(threes)) + ' tweets labeled with a 3'
	print 'smallest list is of size' + str(smallest)

	#shuffling
	#random.shuffle(ones)
	#random.shuffle(twos)
	#random.shuffle(threes)

	#trimming
	ones = ones[:smallest]
	twos = twos[:smallest]
	threes = threes[:smallest]

	#concatenating
	tweets_and_labels.extend(ones)
	tweets_and_labels.extend(twos)
	tweets_and_labels.extend(threes)

	#random.shuffle(tweets_and_labels)
	return tweets_and_labels

def normalize(tweet): 
	# get rid of certain punctuation chars
	symbols_to_eliminate = ['.', '-', ',']
	for symbol in symbols_to_eliminate:
		tweet.replace(symbol, '')

	toks = nltk.word_tokenize(tweet)

	# only take words - things with lowercase letters 
	toks = [w.lower() for w in toks]

	return toks

def ngrams(iterable, n=1):
	l = len(iterable)
	for idx in range(l):
		if idx + n < l : yield iterable[idx:idx+n]

#returns all n grams in toks
def ngram_features(toks, n=1) : 
	n_dict = {}
	for i in range(1,n+1):
		n_dict.update({str(w) : 1 for w in ngrams(toks,i)})
	#print n_dict
	return n_dict

def get_features(data) :
	feat = []
	for tweet in data:
		toks = normalize(tweet)
		#print toks
		tweet_feat = ngram_features(toks, 2)
		feat.append(tweet_feat)
	feats = dv.fit_transform(feat)
	return feats

def get_x_y(data):
	le.fit(sentiments)
	#print data
	Y = le.transform([d[1] for d in data])
	X = get_features([d[0] for d in data])
	print "Y, X SIZE", len(Y)
	return Y, X

def min_sparse(X):
    if len(X.data) == 0:
        return 0
    m = X.data.min()
    return m if X.getnnz() == X.size else min(m, 0)

def print_top_features(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top20 = np.argsort(clf.coef_[i])[-20:]
        print("%s: %s" % (class_label, " ".join(feature_names[j] for j in top20)))
        print("\n")

filename = "../training_data/ordered_tweets_no_duplicates.txt"
tweets_and_labels = parse_labeled_data(filename)
#print tweets_and_labels
#random.shuffle(tweets_and_labels)
Y, X = get_x_y(tweets_and_labels)
#X, Y = make_moons(noise=0.3, random_state=0)
#print X, Y
#print nX[0], nY[0]
#splitting training and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

#C = regularization parameter (keeps from overfitting): C is the degree of penalty (L1 or L2) (powers of 10)
#penalty sparse = l2 lowers angle so that no unigram can be super weighted, l1 removes features to shift the curve
#TODO: separate into train test eval

fs = SelectFwe(alpha=700.0)
print "Before", x_train.shape

clf = svm.LinearSVC(C=100, penalty='l2', dual = False)
clf.fit(x_train, y_train)

print "NO FEATURE SELECTION"
print "Training Accuracy"
print clf.decision_function(x_train)
print (classification_report(y_train, clf.predict(x_train), target_names=target_names))

print "Testing Accuracy"
print (classification_report(y_test, clf.predict(x_test), target_names=target_names))




x_train = fs.fit_transform(x_train, y_train)

print "After", x_train.shape

clf.fit(x_train, y_train)
'''
w = clf.coef_
print w
a = np.array(w[0].todense(), dtype=np.float)
b = np.array(w[1].todense(), dtype=np.float)
c = -100*a/b
print a, b, c
xx = np.linspace(-5, 5)
yy = c * xx - clf.intercept_[0] / b


# get the separating hyperplane using weighted classes
wclf = svm.SVC(kernel='linear', class_weight={1: 10})
wclf.fit(x_train, y_train)

ww = wclf.coef_[0]
wa = -ww[0] / ww[1]
wyy = wa * xx - wclf.intercept_[0] / ww[1]

# plot separating hyperplanes and samples
h0 = pl.plot(xx, yy, 'k-', label='no weights')
h1 = pl.plot(xx, wyy, 'k--', label='with weights')
pl.scatter(X[:, 0], X[:, 1], c=y, cmap=pl.cm.Paired)
pl.legend()

pl.axis('tight')
pl.show()
'''

'''# the function that I'm going to plot
def z_func(x,y):
	return (1-(x**2+y**3))*exp(-(x**2+y**2)/2)
 
x = arange(-3.0,3.0,0.1)
y = arange(-3.0,3.0,0.1)
X,Y = meshgrid(x, y) # grid of point
Z = z_func(X, Y) # evaluation of the function on the grid

im = imshow(Z,cmap=cm.RdBu) # drawing the function
# adding the Contour lines with labels
cset = contour(Z,arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
colorbar(im) # adding the colobar on the right
# latex fashion title
title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
show()


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                      cmap=cm.RdBu,linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
'''
'''
clf.fit(x_train, y_train)
w = clf.coef_.todense()

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
cset = ax.contour(X, Y, Z, 16, extend3d=True)
ax.clabel(cset, fontsize=9, inline=1)
plt.show()
'''
#print_top_features(dv, clf, target_names)
#m = coo_matrix(w)
#print m
#ax = plot_coo_matrix(m)
#ax.figure.show()


# get the separating hyperplane
print "WWWWWWWW", np.array(clf.coef_)
w = np.array(clf.coef_)
print "TESTING", w[:,0]
print np.min(w[:,0])
print clf.intercept_[0]
#print clf.get_params(True)
a = np.divide(-w[:,0], w[:,1])
print a.shape
xx = np.linspace(-5, 5)
print xx.shape, clf.intercept_[0], #clf.support_vectors_
yy = np.dot(a, xx) - np.divide(clf.intercept_[0] ,w[:,1])
print yy

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=80, facecolors='none')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()


print "Training Accuracy"
#print clf.decision_function(x_train)
print (classification_report(y_train, clf.predict(x_train), target_names=target_names))
x_test = fs.transform(x_test)

print "Testing Accuracy"
print (classification_report(y_test, clf.predict(x_test), target_names=target_names))
decisions = clf.decision_function(x_test)
print "DECISION", decisions.shape[1]
#print y_test
X = np.array(decisions[:,0])
#print X
Y = np.array(decisions[:,2])
Z = np.array(decisions[:,1])
points = []
for i, val in enumerate(X):
	#print X[i], Y[i], Z[i]
	points.append((X[i], Y[i], Z[i]))
points = list(set(points))
print points, len(points)
#print X, Y, Z
new_y = []
print "Y_TEST", len(y_test)
for i, val in enumerate(y_test):
	if val == 2:
		val = 'b'
		mark = 'o'
	elif val == 1:
		val = 'r'
		mark = '+'
	else:
		val = 'g'
		mark = '^'
	#print new_y
	new_y.append((val, mark))
#print new_y
#print np.array(X)
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, val in enumerate(np.array(X)):
	ax.scatter3D(X[i], Y[i], Z[i], c=new_y[i][0], marker=new_y[i][1])

ax.set_xlabel('Self')
ax.set_ylabel('General Disease')
ax.set_zlabel('Another Person')
ax.set_autoscale_on(True)

plt.show()
'''

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
for i, val in enumerate(np.array(X)):
	print val
	if new_y[i][0] != 'b': 
		ax2.scatter(X[i], Z[i], c=new_y[i][0], marker=new_y[i][1])
ax2.set_xlabel('Self')
ax2.set_ylabel('Another Person')
ax2.set_autoscale_on(True)

plt.show()

'''
# Define the points in 3D space
# including color code based on Z coordinate.
pts = mlab.points3d(X, Y, Z, Z)

# Triangulate based on X, Y with Delaunay 2D algorithm.
# Save resulting triangulation.
mesh = mlab.pipeline.delaunay2d(pts)

# Remove the point representation from the plot
pts.remove()

# Draw a surface based on the triangulation
surf = mlab.pipeline.surface(mesh)

# Simple plot.
mlab.xlabel("x")
mlab.ylabel("y")
mlab.zlabel("z")
mlab.show()
'''

'''h = .02  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
#svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)
lin_svc = svm.LinearSVC(C=C).fit(X, Y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel',
          'LinearSVC (linear kernel)']


for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    pl.subplot(2, 2, i + 1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pl.contourf(xx, yy, Z, cmap=pl.cm.Paired)
    pl.axis('off')

    # Plot also the training points
    pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

    pl.title(titles[i])

pl.show()
'''