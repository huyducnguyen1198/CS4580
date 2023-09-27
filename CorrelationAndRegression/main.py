import pandas as pd
import plotly.express as px
import numpy as np
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

df = pd.read_csv('weatherAUS.csv')



def printSep(s='-'):
    print()
    print(f"{'-' * 25}{s}{'-' * (50 - len(s))}")
    print()


def printHeader(s="Header"):
    print('*' * 75)
    print(f"{'*' * 25}{s}{'*' * (50 - len(s))}")
    print('*' * 75)


def test(df):
    df['Year'] = df['Date'].apply(lambda x: x[:4])
    df['Day'] = "0000-" + df['Date'].apply(lambda x: x[5:])
    gby = df.groupby('Day')
    fig = px.scatter(x=df['Day'].unique(), y=gby['MaxTemp'].mean())
    fig.show()


def BasicInfo(df):
    ################## Basic Info ###########################3
    print("Basic Infomation")
    ## First Dot
    printSep(1)
    df['ConvertedDate'] = pd.to_datetime(df['Date'])
    print(f"{'Start Day':^20}{df['ConvertedDate'].min().date()}")
    print(f"{'End Day': ^20}{df['ConvertedDate'].max().date()}")

    ## Second Dot
    printSep(2)
    df['Month'] = df['Date'].apply(lambda x: x[5:7])
    avgMinTemp = df.groupby(df['Month'])['MinTemp'].mean()
    print(f"{'Average MinTemp by' : ^20}{avgMinTemp}")

    ## Third Dot
    printSep("3. fig shown")
    fig = px.bar(x=range(1, 13), y=avgMinTemp)
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Avg Min Temperature in C")

    # fig.show()
    ## Fourth Dot
    printSep("4")
    print(f"{'Number of City': ^20}{len(df['Location'].unique())}")

    # avgRain = -np.sort(-df.groupby(df['Location'])['Rainfall'].mean())
    print()
    avgRain = pd.Series(df.groupby(['Location'])['Rainfall'].mean())
    avgRain = avgRain.sort_values(ascending=False)
    print(f"{'Top five Rainiest by ':^20}{avgRain.head(5)}")

    ## Fifth Dot
    printSep(5)

    print(f"{'Min Pressure9am':^20}{df['Pressure9am'].min()}")
    print(f"{'Avg Pressure9am': ^20}{df['Pressure9am'].mean()}")
    print(f"{'Max Pressure9am':^20}{df['Pressure9am'].max()}")

    print(f"{'Min Cloud9am':^20}{df['Cloud9am'].min()}")
    print(f"{'Avg Cloud9am': ^20}{df['Cloud9am'].mean()}")
    print(f"{'Max Cloud9am':^20}{df['Cloud9am'].max()}")

    print(f"{'Mode Temp':^20}{df['Temp9am'].mode()}")


def firstCorr(df):
    printHeader("Correlation")
    printSep("First Cor")

    # First dot

    corrP = df[['MinTemp', 'Rainfall']].corr(method='pearson')
    print(f"{'Pearson':^20}\n{corrP}")
    corrS = df[['MinTemp', 'Rainfall']].corr(method='spearman')
    print(f"\n{'Spearson':^20}\n{corrS}")

    fig = px.scatter(df, 'MinTemp', 'Rainfall')
    # fig.show()

    print(f"\nThe correlation between MinTemp and Rainfall is {round(corrS['MinTemp'].values[1], 3)} according \n"
          f"to spearman method and {round(corrP['MinTemp'].values[1], 3)} using pearson, Both of which is low. \n"
          f"It indicates that the correlation between these two is not strong")


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
def secCorr(df):
    
    cities = ['Sale', 'SydneyAirport', 'Nuriootpa', 'Mildura','Albury']

    city_df = df[['Date','Location', 'Rainfall']][df['Location'].isin(cities)]
    piv = city_df.pivot(index='Date', columns='Location', values='Rainfall').fillna(0)

    ##get correlation between city and fill diiagonal with nan to find max
    corr = piv.corr()

    np.fill_diagonal(corr.values, np.nan)
    ##get max pair, that is the two city with the most correlation
    maxPair = corr.unstack().idxmax()


    ##linear regression
    xx = piv[maxPair[0]]
    y = piv[maxPair[1]]
    lr = LinearRegression()
    xx = np.reshape(xx, (-1,1))
    lr.fit(xx, y)

    plt.scatter(xx,y)
    plt.plot(xx, lr.predict(xx), color='blue')
    plt.title(f'{maxPair[0]} - {maxPair[1]}')
    plt.xlabel(maxPair[0])
    plt.ylabel(maxPair[1])
    

    ###### answer ######
    print(f'Correlation for five city: \n{corr}\n')

    print("Of all the city, Albury has the most correlation with Mildura as it shows that raining in Albury greatly affects the raining in Mildura.\n"
          "The pair is followed by Albury and Sale. Albury and Nuriootpa only half correlated as Albury-Mildura.\n"
          "Albury-sydneyAirport shows almost no correlation.\n"
          "Other significant correlation is Nuriootpa and Mildura. Sale-Mildura is comparable to Albury-Nuriootpa.\n"
          "Interestingly, SydneyAirport shows a negative correlation to Nuriootpa, which means raining in one may reduce rainign in the other\n"
          "Although the correlation is small.")
    plt.show()




from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
def logitReg(x, y):
    classifier = LogisticRegression()
    classifier.max_iter = 10 ** 6

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    print(confusion_matrix(y_test, y_pred))
    print(f'Accuracy = {str(accuracy_score(y_test, y_pred))}')


def part1(df):
    y = df['RainTomorrow']
    df['RainMorrowLogit'] = y.apply(lambda x: 0 if x == "No" else 1)
    x = df[['Evaporation', 'Rainfall', 'Humidity3pm', 'Humidity9am',
            'Cloud3pm', 'Cloud9am', 'Temp3pm', 'Temp9am']].fillna(0)
    '''
    #plot correlation

    sns.pairplot(x)
    plt.show()'''
    print(f"Feature: {x.columns.values}")

    logitReg(x, y)

from sklearn.feature_selection import RFE
from sklearn. tree import DecisionTreeClassifier


def part2(df):
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
    y = df['RainTomorrow']
    ty = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
          'Sunshine', 'WindSpeed9am', 'WindSpeed3pm',
          'Humidity3pm', 'Humidity9am', 'Pressure3pm',
          'Pressure9am', 'Cloud3pm', 'Cloud9am',
          'Temp3pm', 'Temp9am']
    nFeatures = np.array(ty)
    x = df[nFeatures].fillna(0)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    rfe_fitted = rfe.fit(x, y)
    selected = nFeatures[rfe_fitted.support_]
    print(f"Top 5 Feature: {selected}")

    x = df[selected].fillna(0)
    logitReg(x, y)


# BasicInfo(df)
from sklearn.decomposition import PCA

def part3PCA(df):
    ty = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
          'Sunshine', 'WindSpeed9am', 'WindSpeed3pm',
          'Humidity3pm', 'Humidity9am', 'Pressure3pm',
          'Pressure9am', 'Cloud3pm', 'Cloud9am',
          'Temp3pm', 'Temp9am', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']

    df_onehot = pd.get_dummies(df[ty], columns=['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
    df_onehot['Day'] = df['Date'].apply(lambda x: x[8:])
    df_onehot['Year'] = df['Date'].apply(lambda x: x[:4])
    df_onehot['Month'] = df['Date'].apply(lambda x: x[5:7])

    x = df_onehot.fillna(0)
    #x_scaled = pd.DataFrame(x, columns=x.columns)

    pca = PCA(n_components=18)

    pca_f = pca.fit_transform(x)
    pca_f = StandardScaler().fit_transform(pca_f)

    logitReg(pca_f, df['RainTomorrow'])

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def part3LDA(df):
    ty = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
          'Sunshine', 'WindSpeed9am', 'WindSpeed3pm',
          'Humidity3pm', 'Humidity9am', 'Pressure3pm',
          'Pressure9am', 'Cloud3pm', 'Cloud9am',
          'Temp3pm', 'Temp9am', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']

    df_onehot = pd.get_dummies(df[ty], columns=['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
    df_onehot['Day'] = df['Date'].apply(lambda x: x[8:])
    df_onehot['Year'] = df['Date'].apply(lambda x: x[:4])
    df_onehot['Month'] = df['Date'].apply(lambda x: x[5:7])

    x = df_onehot.fillna(0)
    x = StandardScaler().fit_transform(x)
    lda = LinearDiscriminantAnalysis()


    lda_f = lda.fit_transform(x, df['RainTomorrow'])


    logitReg(lda_f, df['RainTomorrow'])

from sklearn.neural_network import MLPClassifier

def MLP(df):
    ty = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
          'Sunshine', 'WindSpeed9am', 'WindSpeed3pm',
          'Humidity3pm', 'Humidity9am', 'Pressure3pm',
          'Pressure9am', 'Cloud3pm', 'Cloud9am',
          'Temp3pm', 'Temp9am']
    ty1 = ty.copy()
    ty1.append('Location')
    df_onehot = pd.get_dummies(df[ty1], columns=['Location'])
    x = df_onehot.fillna(0)
    x = StandardScaler().fit_transform(x)
    y = df['RainTomorrow']

    # PCA 5 componet
    pca = PCA(n_components=15)
    x = pca.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

    #x_scaled = pd.DataFrame(x, columns=x.columns)

    mlp = MLPClassifier(solver='adam', alpha=.005, hidden_layer_sizes=(150,150), learning_rate='constant',batch_size=1024,  verbose=True, max_iter=20)
    mlp.fit(x_train, y_train)

    # Record training history (accuracy and loss) during training

    # Evaluate the model on the test set
    y_pred_test = mlp.predict(x_test)

    y_pred = mlp.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(f'Accuracy = {str(accuracy_score(y_test, y_pred))}')
    print(mlp.score(x_train, y_train))

'''
from tensorflow import keras

def DNN(df):
    ty = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
          'Sunshine', 'WindSpeed9am', 'WindSpeed3pm',
          'Humidity3pm', 'Humidity9am', 'Pressure3pm',
          'Pressure9am', 'Cloud3pm', 'Cloud9am',
          'Temp3pm', 'Temp9am','Location','WindGustDir','WindDir9am', 'WindDir3pm']

    ty1 = ty.copy()
    ty1.append('Location')
    df_onehot = pd.get_dummies(df[ty1], columns=['Location','WindGustDir','WindDir9am', 'WindDir3pm'])
    x = df_onehot.fillna(0)



    y = df['RainTomorrow']
    df['RainMorrowLogit'] = y.apply(lambda x: 0 if x == "No" else 1)


    pca = PCA(n_components=15)
    x = pca.fit_transform(x)
    x = x.reshape(x.shape[0], x.shape[1], 1)

    print(x.shape[1])
    y = df['RainMorrowLogit']
    model = keras.Sequential([
        keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(x.shape[1], 1)),
        keras.layers.Conv1D(filters=256, kernel_size=2, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')

    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(x, y, epochs=18, batch_size=128, validation_split=0.15)

    
    y = df['RainMorrowLogit']

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=len(x[0],)))
    model.add(keras.layers.Dense(units=512, activation='relu'))

    model.add(keras.layers.Dense(units=256, activation='relu'))
    model.add(keras.layers.Dense(units=128, activation='relu'))

    model.add(keras.layers.Dense(units=64, activation='relu'))
    model.add(keras.layers.Dense(units=32, activation='relu'))

    model.add(keras.layers.Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    ## input
    history = model.fit(x, y, epochs=18, batch_size=128, validation_split=0.15)


    ## input

    fig, axs = plt.subplots(1,2)

    axs[0].plot(history.history['accuracy'], label='Training Accuracy')
    axs[0].plot(history.history['val_accuracy'], label='Testing Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    axs[1].plot(history.history['loss'], label='Training loss')
    axs[1].plot(history.history['val_loss'], label='Testing loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('loss')
    axs[1].legend()
    plt.show()

    model.save('test.keras')
'''


def EDA(df):
    print(df.isna().sum())  # check missing value
    px.imshow(df.select_dtypes(include='number').corr(),labels=dict(x="Features", y="Features", color="Correlation"),template='plotly_dark').show()


    '''check distribution of each feature
    df_num = df.select_dtypes(include=np.number)
    
    colors = ['#7DBCE6', '#EEBDEE', '#EAEAAF', '#8FE195', '#E28181',
              '#87D8DB', '#C2E37D', '#DF93A4', '#DCB778', '#C497DE']


    def plot_distribution(df_num, var):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(df[var], color="blue", kde=True)
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[var], color="green", orient="v", width=0.5, linewidth=4)
        plt.show()
        var = df[var]
        varvalue = var.value_counts()
    
        print("{}: \n {}".format(var.name, varvalue))
    
    
    for col in df_num.columns:
        plot_distribution(df_num, col)'''




import xgboost as xgb
from sklearn.metrics import mean_squared_error, classification_report
import graphviz
def XGBoost3(df):
    ty = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
          'Sunshine', 'WindSpeed9am', 'WindSpeed3pm',
          'Humidity3pm', 'Humidity9am', 'Pressure3pm',
          'Pressure9am', 'Cloud3pm', 'Cloud9am',
          'Temp3pm', 'Temp9am', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']

    df_onehot = pd.get_dummies(df[ty], columns=['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
    df_onehot['Day'] = df['Date'].apply(lambda x: x[8:])
    df_onehot['Year'] = df['Date'].apply(lambda x: x[:4])
    df_onehot['Month'] = df['Date'].apply(lambda x: x[5:7])

    x = df_onehot.fillna(0)
    x = StandardScaler().fit_transform(x)

    y = df['RainTomorrow'].apply(lambda x: 0 if x == "No" else 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
    model = xgb.XGBClassifier(objective="reg:squarederror", random_state=42)
    model.fit(x, y)
    y_pred = model.predict(x_test)


    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")

    # Example for classification
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
def plotCorr(df):
    df['RainMorrowLogit'] = df['RainTomorrow'].apply(lambda x: 0 if x == "No" else 1)
    ty = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
              'Sunshine', 'WindSpeed9am', 'WindSpeed3pm',
              'Humidity3pm', 'Humidity9am', 'Pressure3pm',
              'Pressure9am', 'Cloud3pm', 'Cloud9am',
              'Temp3pm', 'Temp9am', 'RainMorrowLogit','RISK_MM']
    corr_df = df[ty].corr()
    '''change plt fig size'''
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_df, annot=True,fmt='.2f', cmap='RdYlGn',annot_kws={'size': 10})
    plt.show()




#DNN(df)
#MLP(df)
#part2(df)
#part1(df)
#secCorr(df)


#part2(df)


XGBoost3(df)
