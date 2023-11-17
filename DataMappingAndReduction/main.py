import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', None)


def printSep():
    print("-" * 120)


df = pd.read_csv('acs2015_census_tract_data.csv')


######## question 1 ########
def q1(df):
    print("Question 1")
    fil = df[np.logical_and(df['State'] != 'Puerto Rico', df['State'] != 'District of Columbia')]

    races = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific']
    racesPop = [x + "Pop" for x in races]

    ##new
    # use loc to assign columns to existing column
    fil.loc[:, races] = fil.loc[:, races]/100

    temp = fil[races].mul(fil['TotalPop'], axis=0)
    #use dicti to assign columns to new columns
    dct = {}
    for i in range(len(races)):
        dct[racesPop[i]] = temp[races[i]]
    fil = fil.assign(**dct)

    ##fji
    gby = fil.groupby('State')
    print(gby.apply(lambda x: x[racesPop].sum()/x['TotalPop'].sum())[racesPop].idxmax())
    printSep()

    # for question 2
    def mul(x):
        return x['TotalPop'] * x['Unemployment'] /100
    fil['UnEmpPop'] = fil[['TotalPop', 'Unemployment']].apply(mul, axis=1)

    print("Question 2")
    gby = fil.groupby('State')
    print(gby.apply(lambda x: x['UnEmpPop'].sum()/ x['TotalPop'].sum()).agg(LowestUnemp='idxmin', HighestUnemp='idxmax')    )
    printSep()
    #q2(gby)


######## question 2#######
def q2(gby):
    print("Question 2")

    print(gby['Unemployment'].mean().agg(LowestUnemp='idxmin', HighestUnemp='idxmax'))
    printSep()


'''
report filter function for question 3, 4, 5
print State, County, hispanic, White, Black, Asian, Native, Pacific that is greater than 1%
args: df
return 
'''


def report(df):
    # tab for table header, which columns to take
    tab = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific']
    header = ['CensusTract', 'State', 'County']

    # tabT for True/False table, which race has more than 1% of pop
    tabT = df[tab].map(lambda x: x > 1)

    # add state and county to table, and True Table, make them both true.(for apply dataframe purpose)
    tab = np.append(header, tab)
    tabT['CensusTract'] = True
    tabT['State'] = True
    tabT['County'] = True

    # apply to dataframe by get columns from tab, and then mask them from tabT, then fill all Nan
    print(df[tab][tabT].fillna(" "))


## question 3
def q3(df):
    print("Question 3")
    fil = df[df['Income'] >= 50000]  # average income
    fil = fil[fil['Poverty'] > 50]

    report(fil)
    printSep()

## question 4
def q4(df):
    print("Question 4")

    df['WomanPercentage'] = df['Women'].divide(df['TotalPop'])
    fil = df[df['WomanPercentage'] > .57]
    fil = fil[fil['TotalPop'] >= 10000]

    report(fil)


## question 5
def q5(df):
    printSep()
    print("Question 5")
    # find four race each of which is more than 15% pop
    races = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific']

    # number of races that have at least 15% of total pop
    df['NumRace15'] = np.sum(df[races] >= 15, axis=1)

    fil = df[df['NumRace15'] >= 4]

    report(fil)


############################ main  ######################################################
print("Huy Nguyen")

q1(df)

q3(df)
q4(df)
q5(df)
#print(df.groupby('State')['TotalPop'].sum())
