print(__doc__)

import numpy as np
import pylab as pl
import nltk
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
import random, re, collections, itertools

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

figure = pl.figure(figsize=(27, 9))
i = 1
# iterate over datasets
#for ds in datasets:
# preprocess dataset, split into training and test part



sentiments = [1 ,2, 3]
target_names = ["Self", "Another Person", "General Statement"]
dv = DictVectorizer()
le = LabelEncoder()

def removeNonAscii(s): return "".join(i for i in s if ord(i)<128)

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
    random.shuffle(ones)
    random.shuffle(twos)
    random.shuffle(threes)

    #trimming
    ones = ones[:smallest]
    twos = twos[:smallest]
    threes = threes[:smallest]

    #concatenating
    tweets_and_labels.extend(ones)
    tweets_and_labels.extend(twos)
    tweets_and_labels.extend(threes)

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
    return n_dict

def get_features(data) :
    feat = []
    for tweet in data:
        toks = normalize(tweet)
        tweet_feat = ngram_features(toks, 2)
        feat.append(tweet_feat)
    feats = dv.fit_transform(feat)
    return feats

def get_x_y(data):
    le.fit(sentiments)
    #print data
    Y = le.transform([d[1] for d in data])
    X = get_features([d[0] for d in data])
    #print "Y, X SIZE", len(Y), Y, X
    return Y, X

def print_top_features(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label, " ".join(feature_names[j] for j in top10)))
        print("\n")

filename = "../training_data/ordered_tweets_no_duplicates.txt"
tweets_and_labels = parse_labeled_data(filename)
















def min_sparse(X):
    if len(X.data) == 0:
        return 0
    m = X.data.min()
    print m
    return m if X.getnnz() == X.size else min(m, 0)

def max_sparse(X):
    if len(X.data) == 0:
        return 0
    m = X.data.max()
    print m
    return m if X.getnnz() == X.size else max(m, 0)

def meshgrid2(*arrs):
    arrs = tuple(reversed(arrs))  #edit
    lens = map(len, arrs)
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz*=s

    ans = []    
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j!=i:
                arr2 = arr2.repeat(sz, axis=j) 
        ans.append(arr2)

    return tuple(ans[::-1])





y, X = get_x_y(tweets_and_labels)
#print X,y
#X, y = make_moons(noise=0.3, random_state=0)
#print X,y

figure = pl.figure(figsize=(27, 9))
i = 1

#X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
h = 0.02
clf = svm.SVC(kernel='linear', C=100)
clf.fit(X_train, y_train)

x_min, x_max = min_sparse(X[:, 0]) - 1, max_sparse(X[:, 0]) + 1
y_min, y_max = min_sparse(X[:, 1]) - 1, max_sparse(X[:, 1]) + 1
z_min, z_max = min_sparse(X[:, 2]) - 1, max_sparse(X[:, 2]) + 1
xx, yy, zz = meshgrid2(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h),
                     np.arange(z_min, z_max, h))
print zz.ravel()

A = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])


pl.contourf(xx, yy, zz, A, cmap=pl.cm.Paired)

'''# just plot the dataset first
cm = pl.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = pl.subplot(len(datasets), len(classifiers) + 1, i)
# Plot the training points
print X_train
ax.scatter(X_train, X_train, c=y_train, cmap=cm_bright)
# and testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
i += 1

# iterate over classifiers
for name, clf in zip(names, classifiers):
    ax = pl.subplot(len(datasets), len(classifiers) + 1, i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    i += 1

figure.subplots_adjust(left=.02, right=.98)'''
pl.show()