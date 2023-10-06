import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2 as c2
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder

from statsmodels.formula.api import ols
from statsmodels.stats.multicomp  import pairwise_tukeyhsd, MultiComparison
import statsmodels.api as sm
# set up pandas to display all columns
pd.set_option('display.max_columns', False)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)


def readFile():
    df = pd.read_csv('Titanic Crew.csv')
    return df


df = readFile()
df = df.rename(columns={'Class/Dept': 'Class_Dept'})
############filter by survived? column
# -1 = Nan, 0 = LOST, 1 = SAVED
df['Survived?'].fillna(-1, inplace=True)

#filter df by 1 and 0
df = df[df['Survived?'] != -1]

#############gender
def chi2Congin(df, indep, dep, alpha=0.05):

    '''Chi2 Test for Independence'''

    print("Hyphothesis Testing")
    '''print Null Hypothesis: there is no significant difference between gender and survival rate'''
    H0 = f"There is no association between {indep} and {dep}"
    Ha = f"There is association between {indep} and {dep}"
    print("H0:", H0)
    print("Ha:", Ha)
    print('.' * 70)
    cong_df = pd.crosstab(df[indep], df[dep])
    chi2Stat, p, dof, expected = chi2_contingency(cong_df)
    chi2Crit = c2.ppf(1-p, dof)
    print(cong_df)

    print('.'*70)
    print(f"{'Chi2_stat': <20} {'p-value': <20} {'alpha':<7} {'dof': <3} {'Chi2_crit': <20}")
    print(f"{chi2Stat: <20} {p: <20} {.05: <7} {dof: <3} {chi2Crit: <20}")
    print('.'*70)
    if p < 0.05:
        print("Reject Null Hypothesis")
        print(f"It is statistically significant to support that \n{Ha}")
    else:
        print("Fail to Reject Null Hypothesis")
        print(f"It is not statistically significant to support that \n{H0}")
    print('.'*70)

def Anova(df):
    print(df['Age'])
    model = ols("Age ~ Class_Dept", data=df).fit()
    aov_table = sm.stats.anova_lm(model)
    print(aov_table)
    mc = MultiComparison(df['Age'], df['Class_Dept'])
    result = mc.tukeyhsd()
    print(result)

def SeparateGender(df):
    ### perfrom chi2 on gender and surviced
    '''df_gen = df[['Gender', 'Survived?']]
    cross = pd.crosstab(df_gen['Gender'], df_gen['Survived?'], margins=True)
    chi2Stat, p, dof, expected = chi2_contingency(cross)
    chi2Crit = c2.ppf(1 - p, dof)
    plot = False
    if plot:
        cross_noMarg = cross.iloc[:-1, :-1]
        ax = cross_noMarg.plot(kind='bar', figsize=(10, 6), stacked=True)
        ax.set_xlabel("Categories")
        ax.set_ylabel("Count")
        plt.show()
    else:
        print(cross)

    print('.' * 70)
    print(f"{'Chi2_stat': <20} {'p-value': <20} {'alpha':<7} {'dof': <3} {'Chi2_crit': <20}")
    print(f"{chi2Stat: <20} {p: <20} {.05: <7} {dof: <3} {chi2Crit: <20}")
    print('.' * 70)'''

    ### get correlation between gender and survived
    df_corr = df[['Gender', 'Survived?']]
    df_corr = pd.get_dummies(df_corr, columns=['Gender']).apply(LabelEncoder().fit_transform)
    print(df_corr.corr())

def printSep(indep, dep):
    s = f" {indep} and {dep}"
    print()
    print("*" * 70)
    print("*"*(35-len(s)//2 - 1), s, "*"*(35-len(s)//2 - 1))
    print("*" * 70)
    print()

##############################################

printSep("Chi2 Test for Gender", "Survived?")
chi2Congin(df, 'Gender', 'Survived?')


printSep("Chi2 Test for Class/Dept", "Survived?")
chi2Congin(df, 'Class_Dept', 'Survived?')
#Chi2Test(df)

printSep("Chi2 Test for Joined", "Survived?")
chi2Congin(df, 'Joined', 'Survived?')


printSep("Gender Correlation", "Survived?")
SeparateGender(df)
#############################################


'''fig, ax = plt.subplots(1,3, figsize=(15, 5))

sns.scatterplot(data=df, x='Age', y='Survived?',ax=ax[0])
ax[0].set_title("Age vs Survived?")
sns.scatterplot(data=df, x='Gender', y='Survived?',ax=ax[1])
ax[1].set_title("Gender vs Survived?")
sns.scatterplot(data=df, x='Died', y='Survived?',ax=ax[2])
ax[2].set_title("Died vs Survived?")
plt.tight_layout()'''
#plt.show()
#pairplot
'''
ft = ['Died','Age',  'Survived?']
df_mul = df[ft]
df_mul = pd.concat([df_mul, pd.DataFrame(df['Gender'].map(lambda x:1 if x =='Male' else 0), columns=['Gender'])], axis=1)

sns.pairplot(df_mul, hue='Survived?')
plt.show()'''

print(df.head())

ft = ['Died', 'Age', 'Survived?']
df_mul = df[ft]
print(df.isna().sum())
sns.set(style='darkgrid')
plot = sns.jointplot(data=df_mul, x='Age', y='Died', hue='Survived?', kind='hist')
plot.plot_joint(sns.kdeplot, fill=True, alpha=.5)

plt.show()
