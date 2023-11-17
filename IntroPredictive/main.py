import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

df_train = pd.read_csv('Titanic_crew_train.csv')
df_test = pd.read_csv('Titanic_crew_test.csv')
df_train = df_train.rename(columns={'Class/Dept': 'Class_Dept'})
df_test = df_test.rename(columns={'Class/Dept': 'Class_Dept'})

def classDeptWrangler(x):
    if  "Engineering Crew" in x:
        return "Engineering Crew"
    elif "Deck Crew" in x:
        return "Deck Crew"
    elif "Victualling Crew" in x:
        return "Victualling Crew"
    elif "Restaurant Staff" in x:
        return "Restaurant Staff"
    else:
        return x

df_test['Class_Dept'] = df_test['Class_Dept'].map(classDeptWrangler)
df_train['Class_Dept'] = df_train['Class_Dept'].map(classDeptWrangler)

def getdata(df_train, df_test):
    ft = ['Age', 'Gender', 'Class_Dept', 'Survived?']
    df_train = df_train[ft].dropna()
    x = df_train.drop('Survived?', axis=1)
    x_train = x.apply(LabelEncoder().fit_transform)
    y_train = df_train['Survived?']

    #############test data
    df_test = df_test[ft].dropna()
    x_test = df_test.drop('Survived?', axis=1)
    x_test = x_test.apply(LabelEncoder().fit_transform)
    y_test = df_test['Survived?']

    return x_train, y_train, x_test, y_test
def DecisionTree(df_train, df_test):
    #linear discriminant analysis
    dt = DecisionTreeClassifier(random_state=0)
    #get data
    x_train, y_train, x_test, y_test = getdata(df_train, df_test)

    #fit decision tree
    dt = dt.fit(x_train, y_train)
    #predict
    y_pred = dt.predict(x_test)

    #print results
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
    print('Classification Report: \n', classification_report(y_test, y_pred))
    conf_mat = confusion_matrix(y_test, y_pred)

    return conf_mat[0][0] + conf_mat[1][1]

def DeepLearning(df_train, df_test):
    x_train, y_train, x_test, y_test = getdata(df_train, df_test)

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32),  max_iter=300, random_state=0)
    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(x_test)

    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
    print('Classification Report: \n', classification_report(y_test, y_pred))
    conf_mat = confusion_matrix(y_test, y_pred)
    return conf_mat[0][0] + conf_mat[1][1]


#########################age prediction models #############################
def getDataAgeModel(df_train, df_test):
    ft = ['Age', 'Gender', 'Class_Dept', 'Survived?']
    df_train = df_train[ft].dropna()
    x = df_train.drop('Age', axis=1)
    x_train = x.apply(LabelEncoder().fit_transform)
    y_train = df_train['Age']

    #############test data
    df_test = df_test[ft].dropna()
    x_test = df_test.drop('Age', axis=1)
    x_test = x_test.apply(LabelEncoder().fit_transform)
    y_test = df_test['Age']


    return x_train, y_train, x_test, y_test

def LinearReg(df_train, df_test):
    x_train, y_train, x_test, y_test = getDataAgeModel(df_train, df_test)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    #print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
    return mean_squared_error(y_test, y_pred)
def DecisionTreeAge(df_train, df_test):
    x_train, y_train, x_test, y_test = getDataAgeModel(df_train, df_test)
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    #print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
    return mean_squared_error(y_test, y_pred)
def DeepLearningAge(df_train, df_test):
    x_train, y_train, x_test, y_test = getDataAgeModel(df_train, df_test)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64,32), max_iter=500, random_state=0)
    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(x_test)

    #print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
    return mean_squared_error(y_test, y_pred)

def main():
    print("****************Survival Prediction****************")
    #print reuslt using fstring and the total of 178
    print(f"Decision Tree: {DecisionTree(df_train, df_test)} / 178")
    print("-"*60)
    print(f"Deep Learning: {DeepLearning(df_train, df_test)} / 178")

    print("****************Age Prediction****************")
    print(f"Decision Tree: {DecisionTreeAge(df_train, df_test)}")
    print(f"Deep Learning: {DeepLearningAge(df_train, df_test)}")
    print(f"Linear Regression: {LinearReg(df_train, df_test)}")

main()