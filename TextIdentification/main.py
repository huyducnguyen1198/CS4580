import sys, os

import matplotlib.pyplot as plt
import numpy
import xgboost
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder

authors = ['Austen', 'Baum', 'Verne']
def readFile(folder='Training'):
    '''
    Read all files in the folder and return a dictionary of books by author
    :param folder:
    :return:
    '''
    bookByAuthor = {}
    bookNames = []
    #check all file in the directory
    for a in authors:
        authorDir = f'{folder}/{a}'
        bookList = os.listdir(authorDir)
        books = []
        for b in bookList:
            with open(f'{authorDir}/{b}', 'r') as f:
                books.append(f.read())
                bookNames.append(b)

        bookByAuthor[a] = books
    return bookByAuthor, bookNames

def preprocessData(books, tfidf=None):
    '''
    transform all book into tfidf vector

    :return:
        x: book tfidf vector
        y: author in order of x
        tfidf: tfidf vectorizer fit on x
    '''


    tfidfBooks = {}
    x = []
    y = []

    for a in authors:
        x.extend(books[a])
        y.extend([a] * len(books[a]))

    if tfidf is None:
        tfidf = TfidfVectorizer()
        x = tfidf.fit_transform(x).toarray()
    else:
        x = tfidf.transform(x).toarray()
    return x, y, tfidf

def predictBooks(books, tfidf, model):
    '''
    predict the author of the book
    :param book: book to predict
    :param tfidf: tfidf vectorizer fitted on training data
    :param model: model to predict
    '''
    for a in authors:
        for b in books[a]:
            pred = model.predict(tfidf.transform([b]).toarray())
            if isinstance(pred[0], numpy.int64):
                print(f'{a}: {authors[pred[0]]}')
            else:
                print(f'{a}: {pred}')

def NativeBayes(x, y):
    clf = MultinomialNB()
    clf.fit(x, y)
    return clf

def SVM(x, y):
    clf = svm.SVC()
    clf.fit(x, y)
    return clf

def xgb(x, y):
    clf = xgboost.XGBClassifier()
    clf.fit(x, y)
    return clf

def randomForest(x, y):
    clf = RandomForestClassifier()
    clf.fit(x, y)
    return clf
def test():
    trainBooks, bookNames = readFile()
    x, y, tfidf = preprocessData(trainBooks)
    y_cat = [authors.index(a) for a in y]


    testBooks, bookNames = readFile('Testing')
    x_test, y_test, _ = preprocessData(testBooks, tfidf)
    y_test = [authors.index(a) for a in y_test]

    print('Testing Native Bayes')
    clf = NativeBayes(x, y)
    predictBooks(testBooks, tfidf, clf)

    print('Testing SVM')
    clf = SVM(x, y)
    predictBooks(testBooks, tfidf, clf)


    print('Testing XGBoost')
    clf = xgb(x, y_cat)
    predictBooks(testBooks, tfidf, clf)


    print('Testing Random Forest')
    clf = randomForest(x, y_cat)
    predictBooks(testBooks, tfidf, clf)

    xPCA = PCA(n_components=2).fit_transform(x_test)
    plt.scatter(xPCA[:, 0], xPCA[:, 1], c=y_test, cmap=plt.cm.Set1)
    plt.show()



test()




