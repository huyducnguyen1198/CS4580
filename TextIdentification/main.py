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
import sys
import re

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

def readBook(nameFile='test.txt'):
    print(f'Reading {nameFile}')
    try:
        with open(nameFile, 'r') as f:
            book = f.read()
            return book
    except:
        print("File not found")
        return None


def extract_author(text):
    # Define the regular expression pattern with multiline option
    pattern = r'^Author: (.+)$'

    # Search the text for the pattern
    match = re.search(pattern, text, re.M)

    # If a match is found, return the author's name
    if match:
        return match.group(1)
    else:
        return 'Author not found'

def training():

    # read the file and get the book and the author
    print("Book Identification by Huy Nguyen\n")
    print("Reading Training Data...\n")
    trainBooks, bookNames = readFile()

    # preprocess the data
    x, y, tfidf = preprocessData(trainBooks)
    y_cat = [authors.index(a) for a in y]

    # train the models
    print("Training Native Bayes...")
    bayes = NativeBayes(x, y)
    print("Training SVM...")
    svm = SVM(x, y)
    print("Training XGBoost...")
    xg = xgb(x, y_cat)
    print("Training Random Forest...\n")
    rf = randomForest(x, y_cat)

    # read the past point
    key = ''
    point = 0
    try:
        with open('point.txt', 'r') as f:
            point = int(f.read())
    except:
        pass

    # read the book and predict the author
    while key != 'q':
        key = input("Enter the name of the book to predict or q to quit: ")
        book = readBook(key)
        if book is None:
            continue
        author = extract_author(book)

        #print author fromt the list of authors inform Authors: 1. name, 2. name, 3. name
        print(f'{"Authors:":>15}', end=' ')
        for i, a in enumerate(authors):
            print(f'{i+1}. {a}', end=', ')
        print()
        choice = -1
        while choice < 0 or choice > len(authors):
            choice = int(input("Enter the number of the author: "))
        guessAuthor = authors[choice-1]

        # predict the author
        bayes_pred = bayes.predict(tfidf.transform([book]).toarray())[0]
        svm_pred = svm.predict(tfidf.transform([book]).toarray())[0]
        xg_pred = authors[xg.predict(tfidf.transform([book]).toarray())[0]]
        rf_pred = authors[rf.predict(tfidf.transform([book]).toarray())[0]]

        # print the result
        print(f'{"Algorithm 1:":>15}{"Native Bayes": >20}: {bayes_pred:>15}')
        print(f'{"Algorithm 2:":>15}{"SVM":>20}: {svm_pred:>15}')
        print(f'{"Algorithm 3:":>15}{"XGBoost":>20}: {xg_pred:>15}')
        print(f'{"Algorithm 4:":>15}{"Random Forest":>20}: {rf_pred:>15}')
        print(f'{"Guess author":>35}: {guessAuthor:>15}')
        print(f'{"Real author":>35}: {author:>15}')

        # check if the guess author is correct
        if guessAuthor.lower() in author.lower():
            point += 1
            print("Correct!")
            print(f'{"Point":>35}: {point:>15}')

    with open('point.txt', 'w') as f:
        f.write(f'{point}')
training()




