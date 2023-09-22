import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib as plt
from scipy.stats import chi2 as c2
from scipy.stats import chi2_contingency

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


def printSep(indep, dep):
    s = f"Chi2 Test for {indep} and {dep}"
    print()
    print("*" * 70)
    print("*"*(35-len(s)//2 - 1), s, "*"*(35-len(s)//2 - 1))
    print("*" * 70)
    print()

##############################################

printSep("Gender", "Survived?")
chi2Congin(df, 'Gender', 'Survived?')


printSep("Class/Dept", "Survived?")
chi2Congin(df, 'Class/Dept', 'Survived?')
#Chi2Test(df)

printSep("Joined", "Survived?")
chi2Congin(df, 'Joined', 'Survived?')