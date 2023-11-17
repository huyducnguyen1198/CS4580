import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def printSeparator():
    print()
    print("-" * 60)
    print()


def firstEight(df):
    print("2. First five rows")
    print(df['Temperature (C)'].head())

    printSeparator()
    print("3. Last five rows")
    print(df['Temperature (C)'].tail())
    printSeparator()

    print("4. Min, Max, Std of Temperature (C)")
    print(f"Min: {df['Temperature (C)'].min()}")
    print(f"Max: {df['Temperature (C)'].max()}")
    print(f"Std: {df['Temperature (C)'].std()}")

    '''Use groupby to show the min, max, and standard deviation for the 'Temperature (C)' for the following groups:
        'Clear'
        'Partly Cloudy'
        'Foggy'
    '''
    printSeparator()

    print("5. groupby")
    cond = ["Clear", "Partly Cloudy", "Foggy"]

    fil = df['Summary'] == cond[0]
    fil = np.logical_or(fil, df['Summary'] == cond[1])
    fil = np.logical_or(fil, df['Summary'] == cond[2])

    gb = df[fil].groupby('Summary')
    print(gb.agg({'Temperature (C)': ['mean', 'min', 'max']}))
    # print(df["Temperature (C)"].group.by(by=['Temperature (C)', 'Daily Summary']).agg(['min', 'max']))
    printSeparator()

    print("6. Radians")
    df["Radians"] = df.apply(lambda row: math.radians(row["Wind Bearing (degrees)"]), axis=1)
    print(df["Radians"].describe())
    printSeparator()

    print("7. Filter")
    '''Using filtering, create a new dataframe that only has values where the Humidity is >= 0.6 and <= 0.7.
            With this dataset use the 'describe' function on the 'Wind Speed' column
            With this dataset export the data to an html table with the filename [your last name]_[your first name].html. For example, 'ball_robert.html.'
            '''
    filtered_df = df[(df['Humidity'] >= 0.6) & (df['Humidity'] <= 0.7)]
    print(filtered_df['Wind Speed (km/h)'].describe())
    filtered_df.to_html("Nguyen_Huy.html")


def theRest(df):
    from dateutil.relativedelta import relativedelta

    printSeparator()

    print("8. ")
    convertedDate = pd.to_datetime(df['Formatted Date'], utc=False)

    # cal days, year, moths without lib

    '''   span = convertedDate.max() - convertedDate.min()
    years = span // 365.2425
    months = (span - (years * 365.2425))//30.4
    days = span - int(years*365.2425) - int(months * 30.4)'''

    # cal days  months years with lib

    diff = relativedelta(convertedDate.max(), convertedDate.min())

    print(
        f"     1. Time span: From {convertedDate.min()} To {convertedDate.max()}, in total of {diff.years} years, {diff.months} months, {diff.days} days ")

    print()
    print()

    print("     2. Temperature in C")
    print(df["Temperature (C)"].describe())
    print()
    print(
        f"This place's weather is like most places in USA in term of fluctuation, rather cold(below 0)in winter and high (above 30) in summer")
    print(
        "In term of hotness, the highest temp in Arizona recorded was upto 50 C, which is 10C higher than this place.")
    print(
        "In term of coldnes, the lowest temp in Anchorage recorded was as low as -31C, which is 10C lower than this place. ")
    print("USA's average temp is about 13C, which means this place average is quite close to that of the usa's")

    print()
    print()

    print("     3. Humidity")
    print(df["Humidity"].describe())
    print()

    print("This place is rather humid.")
    print(
        f"This place's humidity varies from 0(dry) to 1(wet). Moreover, its means and std suggest that this is left skewed.")

    print()
    print()
    print("     3. Windy")
    print(df['Wind Speed (km/h)'].describe())
    print()
    print("This place is not so windy")
    print(
        f"This place's wind blows up to 64 kph. However, numbers indicated that the wind speed is right skewed, also its means is lower than American's, which is 19kph. ")

    print(
        "Overall, this place has a slightly wider range of temperature than Utah(slc), it seems to be a touch colder than utah.")
    print("In term of Temperature, this place is almost similar to Utah, at which I am comfortable to live.")
    print("However, There needs more criteria to give a definite answer.")


def getDate(cell):
    return cell[:10]


def plotInfoOverYear(df):
    df["Date"] = df["Formatted Date"].map(getDate)

    df["Year"] = df["Date"].map(lambda cell: cell[:4])
    df["Year"] = pd.to_numeric(df["Year"])
    pTable = pd.pivot_table(df, values="Temperature (C)", index='Year', aggfunc=('min', 'max', 'mean', 'std'))

    fig, axis = plt.subplots(2, 2, figsize=(20, 20))
    axis[0, 0].plot(pTable.index.array, pTable['min'].array)
    axis[0, 0].set_title("min")
    axis[0, 1].plot(pTable.index.array, pTable['max'].array)
    axis[0, 1].set_title("max")
    axis[1, 0].plot(pTable.index.array, pTable['mean'].array)
    axis[1, 0].set_title("mean")
    axis[1, 1].plot(pTable.index.array, pTable['std'].array)
    axis[1, 1].set_title("std")

    plt.show()


def plotDistThruYear(df):
    df["Date"] = df["Formatted Date"].map(getDate)

    df["Year"] = df["Date"].map(lambda cell: cell[:4])
    df['Date'] = pd.to_datetime(df['Date'])

    df["Year"] = pd.to_numeric(df["Year"])

    print(df.head())
    ax = sns.displot(data=df, x='Temperature (C)', hue='Year', kind='kde')
    plt.show()


# canvas do not allow same file name, main.py -> main-1.py
# threfore I remove

# if __name__ == "__main__":

printSeparator()
print("CS4580")
print("Assignment intro to Panda")
print("Huy Nguyen")

printSeparator()
pd.set_option('display.max_columns', None)
df = pd.read_csv('weatherHistory.csv')
firstEight(df)
theRest(df)
