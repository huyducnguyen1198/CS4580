import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
import xgboost as xgb
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', False)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)


def logistAndXgb(df):
    x = df.drop('Species', axis=1)
    y = df['Species']
    x = StandardScaler().fit_transform(x)
    x = PCA(n_components=4).fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # logistic
    print("logixti regression \n")
    logit = LogisticRegression()
    logit.max_iter = 1000
    logit.fit(x_train, y_train)
    y_pred = logit.predict(x_test)

    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print("XGBOost")


    model = xgb.XGBClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")

    # Example for classification
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))


###########read file################
df = pd.read_csv('penguins_no_nulls.csv')

print(df.head())

#df = df.apply(LabelEncoder().fit_transform)

#filter df by type numerical
df_num = df.select_dtypes(include=['float64', 'int64'])
df_num['Species'] = df['Species']

import seaborn as sns
#plot Culmen Length (mm)  Culmen Depth (mm) by Species in seaborn
f = ['Culmen Length (mm)',  'Culmen Depth (mm)',  'Flipper Length (mm)',  'Body Mass (g)','Species']

df_f = df[f]
#df= df.apply(LabelEncoder().fit_transform)

from sklearn.feature_selection import RFE
def rfe(df):
    rfe = RFE(estimator=LogisticRegression(), n_features_to_select=4)
    x = df.drop('Species', axis=1)
    y = df['Species']
    x = StandardScaler().fit_transform(x)
    rfe = rfe.fit(x, y)

    print(rfe.support_)
    print(rfe.ranking_)

    #get feature name from df and rfe
    feature_name = df.drop('Species', axis=1).columns
    print(feature_name)
    print(feature_name[rfe.support_])


topFea = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Body Mass (g)']

df_top = df[topFea]
print(df_top.corr())
## create multiindex

mi = df.set_index(keys = ['Island', 'Species']).sort_values(by=['Island', 'Species', 'Culmen Length (mm)', 'Culmen Depth (mm)'])
#print(mi)


#########

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)
tran = ohe.fit_transform(df[['Island']])
df_is = pd.DataFrame(tran, columns=ohe.get_feature_names_out())

#put df_is into df
df = pd.concat([df, df_is], axis=1)

fea = ohe.get_feature_names_out()
#add Species into fea
fea = list(fea)
fea.append('Species')
df = df[fea]
df = df.apply(LabelEncoder().fit_transform)
print(df.corr())


########
df = pd.read_csv('penguins_no_nulls.csv')
gby = df.groupby(['Island'])['Species'].count()
print(gby)

