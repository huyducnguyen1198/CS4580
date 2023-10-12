import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)





df = pd.read_csv('movies.csv')


#extract genres from a string and remove no genre listed
genre = df['genres'].map(lambda x: x.split('|'))
allGen = np.unique(np.concatenate(genre.values))
allGen = allGen.tolist()
allGen.remove('(no genres listed)')

#mappeing genres to a binary vector each column is a genre
def mapGenre(row):
    return [1 if g in row else 0 for g in allGen]
mappedGen = genre.map(mapGenre)
##binary array to dataframe with column names
mappedGen = mappedGen.apply(pd.Series)
mappedGen.columns = allGen
df = df.join(mappedGen)


#handle text data as in title
tfidf = TfidfVectorizer()
#doing this show that about 75% of the titles are 6 words or less. so use 8
#print(df['title'].map(lambda x: len(x.split(' '))).describe())
def filterTitle(row, n=8):
    words = row.split(' ')
    if len(words) > n:
        return ' '.join(words[:n])
    return row
fil = df['title'].map(filterTitle)
df['title'] = fil
tfidf.fit(df['title'])
transformed = tfidf.transform(df['title'])
#print the list of words in tfidf
print(tfidf.get_feature_names_out())
