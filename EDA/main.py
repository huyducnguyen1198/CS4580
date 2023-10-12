import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.neural_network import MLPClassifier
from statsmodels.formula.api import ols
from statsmodels.sandbox.stats.multicomp import MultiComparison
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import xgboost as xgb
from sklearn.utils import class_weight


pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv('Training Data.csv')
print(df.columns)
ft = ['Income', 'Age', 'Experience','House_Ownership', 'Married/Single', 'Car_Ownership']
x = df[ft]


###########################
''' age vs imcom mean
gby = x.groupby('Age')['Income'].mean()

fig = px.scatter(gby, x=gby.index,y='Income', marginal_y='violin', marginal_x='box')

fig.show()'''
###########################
'''professions vs income mean'''
'''gby = x.groupby('Profession')['Income'].mean()
fig = px.scatter(gby, x=gby.index,y='Income', marginal_y='violin')
fig.show()'''

###########################
'''anova test between age expereicen and income'''
'''model = ols("Income ~ Age * Experience", data=x).fit()
aov_table = sm.stats.anova_lm(model)
print(aov_table)
mc = MultiComparison(x['Income'], x['Age'])
result = mc.tukeyhsd()
print(result[result.pvalues < 0.05])'''
#piv = gby.pivot(index='Age', columns='Married/Single', values='Income')

###########################
'''income between married and single'''''
'''
gby = x.groupby(['Married/Single', 'Age'])['Income'].mean()
gby1 = gby.reset_index()
fig1 = px.scatter(gby1, x='Income', y='Age', color='Married/Single', marginal_x='box')
fig1.show()

data = [gby[gby.index.get_level_values('Married/Single') == 'single'].values, gby[gby.index.get_level_values('Married/Single') == 'married'].values]
group_labels = ['single', 'married']
fig = ff.create_distplot(data, group_labels, bin_size=[50000, 50000])
fig.show()'''

###########################

'''house ownership vs income'''
'''gby = x.groupby(['House_Ownership', 'Age'])['Income'].mean()
gby1 = gby.reset_index()
fig1 = px.scatter(gby1, x='Income', y='Age', color='House_Ownership', marginal_x='box')
fig1.show()

data = [gby[gby.index.get_level_values('House_Ownership') == 'norent_noown'].values, gby[gby.index.get_level_values('House_Ownership') == 'rented'].values, gby[gby.index.get_level_values('House_Ownership') == 'owned'].values]
group_labels = ['norent_noown', 'rented', 'owned']
fig = ff.create_distplot(data, group_labels, bin_size=[50000, 50000, 50000])
fig.show()
'''

###########################
'''house_ownership vs married/single'''
'''subs = make_subplots(rows=1, cols=2)

### house ownership vs married/single
gby = x.groupby(['House_Ownership', 'Married/Single'])['Income'].count()
gby1 = gby.reset_index()
fig1 = px.bar(gby1, x='House_Ownership', y='Income', color='Married/Single', text='Income')
fig1.update_yaxes(title_text='Count')
fig1.update_layout(title_text='House Ownership vs Married/Single')

### car ownership vs married/single
gby2 = x.groupby(['Car_Ownership', 'Married/Single'])['Income'].count()
gby2 = gby2.reset_index()
fig2 = px.bar(gby2, x='Car_Ownership', y='Income', color='Married/Single', text='Income')
fig2.update_yaxes(title_text='Count')
fig2.update_layout(title_text='Car Ownership vs Married/Single')

#add house ownership vs married/single to subplot
subs.add_trace(fig1.data[0], row=1, col=1)
subs.add_trace(fig1.data[1], row=1, col=1)
#add car ownership vs married/single to subplot
subs.add_trace(fig2.data[0], row=1, col=2)
subs.add_trace(fig2.data[1], row=1, col=2)

subs.update_layout( title_text="House/Car Ownership vs Married/Single")
subs.update_xaxes(title_text='Car Ownership ',  row=1, col=2)
subs.update_xaxes(title_text='House Ownership ', row=1, col=1)
subs.show()'''
###########################
'''pca -> logistic regression not working well'''
'''pca = PCA(n_components=2)
x = df.drop(['Id', 'Risk_Flag'], axis=1)
y = df['Risk_Flag']
x = x.apply(LabelEncoder().fit_transform)
x = StandardScaler().fit_transform(x)
x = pca.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = LogisticRegression().fit(x_train, y_train)
disp = DecisionBoundaryDisplay.from_estimator(model, x, response_method='predict',alpha=0.5)
disp.ax_.scatter(x[:, 0], x[:, 1], c=y, cmap='rainbow', edgecolors='k')
plt.show()

y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))'''

###########################
'''correlation matrix -> verry low correlation between features'''

'''ft = ['Married/Single', 'House_Ownership', 'Car_Ownership', 'Profession', 'CITY', 'STATE', 'Risk_Flag']

df_corr = df[ft].apply(LabelEncoder().fit_transform)
df_corr = StandardScaler().fit_transform(df_corr)
df_corr = pd.DataFrame(df_corr, columns=ft)
df_corr = df_corr.corr()

fig = px.imshow(df_corr, color_continuous_scale='RdBu')
fig.show()'''


###########################
'''MLP not good 88% accuracy'''
'''x = df.drop(['Id', 'Risk_Flag'], axis=1)
y = df['Risk_Flag']
x = x.apply(LabelEncoder().fit_transform)
x = StandardScaler().fit_transform(x)
x = PCA(n_components=6).fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = MLPClassifier([ 512,256,256,128,128,64,32], max_iter=200, learning_rate='adaptive',verbose=2).fit(x_train, y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))'''


###########################
'''xgboost'''
clf = xgb.XGBClassifier(max_depth=50, verbosity=2, seed=0)
x = df.drop(['Id', 'Risk_Flag'], axis=1)
y = df['Risk_Flag']
x = x.apply(LabelEncoder().fit_transform)
#########use ballanced class weight to fix the imbalance in the data?




x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#xgboost
'''
classes_weights = class_weight.compute_sample_weight(
    class_weight='balanced',
    y=y_train
)
clf.fit(x_train, y_train, sample_weight=classes_weights)

y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))'''
#ada boost
'''from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=25),n_estimators=100, random_state=0)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))'''

#rus boost random under sampling
'''from imblearn.ensemble import RUSBoostClassifier
from sklearn.tree import DecisionTreeClassifier

rus_boost = RUSBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=25),n_estimators=100, random_state=0)
rus_boost.fit(x_train, y_train)
y_pred = rus_boost.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))'''

#adaboost vs xgboost with smote
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
smote = SMOTE(random_state=0)


print('before smote')
print(x_train.shape)

print(np.sum(y_train == 1), np.sum(y_train == 0))
x_train, y_train = smote.fit_resample(x_train, y_train)
print('after smote')
print(x_train.shape)
print(np.sum(y_train == 1), np.sum(y_train == 0))
#exit()


clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=50),n_estimators=100, random_state=0)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(roc_auc_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#xgboost with smote
clf = xgb.XGBClassifier(max_depth=50, verbosity=2, seed=0)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(roc_auc_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

