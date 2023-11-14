import Levenshtein as lev
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

pd.set_option('expand_frame_repr', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('./movies.csv')


#############################################
#               Preprocessing                #
#############################################

def editIMDB(df):
    def edit(row):
        return 'tt' + str(row).zfill(7)

    df['imdbId'] = df['imdbId'].map(edit)
    return df

    # extract genres from a string and remove no genre listed


def extractGenre(df):
    genre = df['genres'].map(lambda x: x.split('|'))
    allGen = np.unique(np.concatenate(genre.values))
    allGen = allGen.tolist()
    allGen.remove('(no genres listed)')
    return genre, allGen

    # mappeing genres to a binary vector each column is a genre


def transformGenre(df):
    genre, allGen = extractGenre(df)

    def mapGenre(row):
        return [1 if g in row else 0 for g in allGen]

    mappedGen = genre.map(mapGenre)
    ##binary array to dataframe with column names
    mappedGen = mappedGen.apply(pd.Series)
    mappedGen.columns = allGen
    df = df.join(mappedGen)
    return df, allGen

    # extract year from title and add it as a feature and remove year from title
    # used for movies.csv not moviesWRating.csv


def extractYear(df):
    def getYear(row):
        year = row.split('(')[-1].split(')')[0]
        if year.isdigit():
            return int(year)
        return 0

    df['year'] = df['title'].map(getYear)
    df['title'] = df['title'].map(lambda x: x.split('(')[0])
    return df


# use word2vec to transform title into a vector
def word2vec(df):
    # df = extractYear(df)
    titles = df['title'].map(lambda x: x.split(' '))

    model = Word2Vec(titles, min_count=1, vector_size=100, window=2, sg=0)
    model.train(titles, total_examples=len(titles), epochs=100)

    def get_titles_emb(title, model):
        words = title.split()
        emb = [model.wv[word] for word in words if word in model.wv.index_to_key]

        if emb:
            return sum(emb) / len(emb)
        else:
            return [0] * model.vector_size

    emb_title = df['title'].map(lambda x: get_titles_emb(x, model))

    emb_title = emb_title.apply(pd.Series)
    emb_title.columns = ['title_emb_' + str(i) for i in range(100)]
    df = df.join(emb_title)
    df = df.drop(['title', 'genres', 'imdbId', 'movieId', 'year'], axis=1)

    x_train, x_test = train_test_split(df, test_size=0.2, random_state=42)

    '''
    #knn

    #test selecting using genres
    gen = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    #test is an instance of user input searching by genre
    test = x_test[gen].iloc[0]
    #filleer by genre
    fil_by_gen = x_train[gen].apply(lambda x: x * test, axis=1)
    fil_by_gen = fil_by_gen[fil_by_gen.sum(axis=1) != 0]


    #filter x_train and x_test to movies that have at least one genre in common with the test
    fil_index = fil_by_gen.index

    '''
    # do knn

    knn = NearestNeighbors(n_neighbors=100, metric='cosine')
    knn.fit(x_train)
    distances, indices = knn.kneighbors(x_train.loc[0].values.reshape(1, -1))
    check = pd.read_csv('movies.csv')

    idx = x_train.iloc[indices[0]].index.tolist()

    print(check.loc[0][['title', 'genres']])

    print(check.loc[idx][['title']])
    # print(check.loc[indices[0]][['title', 'genres']])


# word2vec(df)

###############################################################
#                   GET KNN MOVIE BASED ON TITLE              #
#                   USING Lavenshtien distance                #
###############################################################

def titlesimilarity(title, df_t=df, n=10):
    '''Calculate similarity between title and all titles in df
        using lavenshtien distance
        :arg:
            - df: dataframe
            - title: string
            - n: number of similar titles to return
        :return:
            - n similar titles location in df(index)
    '''

    def similarity(row, test):
        '''Calculate similarity between two strings using lavenshtien distance
            Typically used for dataframe.apply()
            :arg:
                - row: string
                - test: string
            :return:
                - similarity: float
        '''
        lavenshtienDistance = lev.distance(row, test)
        similarity = 1 - (lavenshtienDistance / max(len(row), len(test)))
        return similarity

    # Calculate lavenshtien similarity for each movie's title
    # Then sort it by similarity descending
    lavenshtien = df_t['title'].apply(lambda x: similarity(x, title))
    firstTen = lavenshtien.sort_values(ascending=False)[:n]
    # firstTenName = df.loc[firstTen.index]['title']
    return df_t.loc[firstTen.index]


############################################################
#                   GET KNN MOVIE BASED ON GENRES          #
#                   USING Jaccard Similarity               #
############################################################

allGen = ['Action', 'Adventure', 'Animation', 'Children',
          'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery',
          'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']


def genresimilarity(df=df, genres=['Drama', 'Comedy'], p=.5):
    '''
    Get a list of similar movies by genre.
    Using jacard similarity
    :param df: dataframe that contain genres column by default a|b|c|d
    :param genres: an array of genres
    :param p: percentage of similarity
    :return: n similar movies location in df(index)
    '''

    def jacard_similarity(row, query):
        '''
        Calculate similarity between two sets using jacard distance
        Typically used for dataframe.apply()
        :arg:
            - row: df row
            - test: set
        :return:
            - similarity: float
        '''

        def getGenre(genre):
            '''
            df default genres is in for a|b|c|d
            this method split it into a list.
            :param genre:
            :return: list of genres:String[]
            '''

            return genre.split('|')

        def createGenreList(genreList):
            '''create and check list of genres using allGen
            :param genreList:String[], a list  string of genres from user input
            :return: a list of official genres
            '''
            l = []
            for g in allGen:
                if g in genreList:
                    l.append(g)
                if g not in allGen:
                    print(f'{g} not in {allGen}')
            return l

        row = set(getGenre(row))
        test = set(createGenreList(query))
        num = len(row.intersection(test))
        den = len(row.union(test))
        return float(num) / float(den)

    # Calculate jaccard similarity for each movie's genres
    # Then sort it by similarity descending
    jaccard = df['genres'].apply(lambda x: jacard_similarity(x, genres))
    genresSortedBySimilarity = jaccard.sort_values(ascending=False)
    # print(genresSortedBySimilarity[:10])
    return df.loc[(genresSortedBySimilarity > p).index]

def yearSimilarity(df, year, n=10):
    return df[df['year'] == year].sample(n)

#################################################################################
#                               SEARCH MOVIE For CLI                            #
#################################################################################

##########################################
#           Search by genre              #
##########################################
def searchByGenre(df, n=10):
    '''
    Search by genre, for CLI
    :param df:
    :param n: number of movies to return
    :return: df of n movies
    '''
    numEle = 5
    for i, g in enumerate(allGen):
        if i % numEle != 0:
            print(f'{i:>3}. {g:<15}', end='')
        else:
            print()

    userIn = input(f'\nPlease enter genres by its index separated by space: ')
    userIn = userIn.split(' ')
    genresIn = [g for i, g in enumerate(allGen) if str(i) in userIn]

    if len(genresIn) == 0:
        return df.sample(n)
    return genresimilarity(df, genresIn)[:n]


###########################################
#               Search by title            #
###########################################
def searchByTitle(df, n=10):
    '''
    Search by title, for CLI
    :param df:
    :param n: number of movies to return
    :return: df of n movies
    '''
    title = input(f'\nPlease enter title: ')
    df_t = df
    return titlesimilarity(title, df_t, n)


###########################################
#       Search by genre then title CLI    #
###########################################

def searchByGenreThenTitle(df, n=10):
    '''
    Search by genre then title, for CLI
    :param df:
    :param n: number of movies to return
    :return: df of n movies
    '''
    dfGen = searchByGenre(df, n=200)
    dfGenTit = searchByTitle(dfGen)

    return dfGenTit[:n]



# Use the function
##########################################################
#                   Get Recommendation                   #
#                   Based on past movies                 #
##########################################################

def getWeightForJaccard(arr=[]):
    '''
    Get weight for jaccard distance based on past movies,
    :param arr: array of list of genres [['Action', 'Children', etc.][][]]
    :return: weight: dict of genres and its weight(count) index is the same as allGen [0,1,...]
    '''
    if len(arr) == 0:
        return None
    weight = {'Total': 0}
    for m in arr:
        for g in m:
            if g in weight:
                weight[g] += 1
            else:
                weight[g] = 1
            weight['Total'] += 1
    return weight


def getMovieRec(df_rec=df, n=10, random_state=0):
    '''
    Get recommendation based on past movies
    :param df_rec:
    :param n: number of movies to return
    :param random_state:
    :return: knn movies to past movies
    '''


    # Load and check past movies

    pastDf = getPastMovies()
    if pastDf is None:
        return df_rec.sample(n, random_state=random_state)
    weight = getWeightForJaccard(pastDf['genres'].map(lambda x: x.split('|')).tolist())
    if weight is None:
        return df_rec.sample(n, random_state=random_state)

    # Remove past movies from df_rec
    # so that it won't recommend past movies
    df_rec = df_rec.drop(pastDf.index)

    # Get weighted_jaccard_similarity for past genres
    # Using weight from getWeightForJaccard()(count of each genres in past movies)
    # Then calculate weighted jaccard similarity using getJaccardSim()
    # Finally, add weighted jaccard similarity to df_rec
    def getJaccardSim(row, weight):
        ''' Get weighted jaccard similarity
        :param row: row of dataframe
        :param weight: dict of genres and its weight(count) index is the same as allGen {'Action': 1, 'Adventure': 2, etc.}
        :return: weighted jaccard similarity
        '''

        row = row.split('|')
        numerator = 0
        denominator = weight['Total']
        for g in row:
            if g in weight:
                numerator += weight[g]
        return numerator / denominator

    jaccard = df_rec['genres'].apply(lambda x: getJaccardSim(x, weight))
    df_rec['jaccard'] = jaccard

    # Get movied based on tfidf-cosine similarity on past title
    # First get tfidf vectorizer of all titles
    # Then combine all past titles into one string
    #       and transform past titles into tfidf vector
    # Final, Calculate cosine similarity between past titles
    #       and all titles via tfidf vector
    tfidf = TfidfVectorizer()
    tfidf.fit(df_rec['title'])
    '''pastDf_vector = tfidf.transform(pastDf['title'])
    cosine_similarities = (cosine_similarity(df_vector, pastDf_vector) + 1)/2.0
    df_rec['cosine'] = cosine_similarities.mean(axis=1)'''
    pastTit = ' '.join(pastDf['title'].tolist())
    pastTit_vector = tfidf.transform([pastTit])

    df_vector = tfidf.transform(df_rec['title'])
    df_rec['cosine'] = cosine_similarity(df_vector, pastTit_vector)

    #  Combine jaccard and cosine similarity
    #  using 20% jaccard and 80% cosine
    df_rec['JacCosScore'] = 0.2 * df_rec['jaccard'] + 0.8 * df_rec['cosine']
    return df_rec.sort_values(by='JacCosScore', ascending=False)[:n]


##########################################
#           Get Recommendation           #
#           Based on Year                #
##########################################

def getYearDif(year1, year2):
    '''
    Get difference between two years
    :param year1:
    :param year2:
    :return: difference
    '''
    return abs(year1 - year2)
def getMovieRecByYear(df_rec=df, yearRef=2020, n=10, random_state=0):
    '''
    Get recommendation based on past movies
    :param df_rec:
    :param n: number of movies to return
    :param random_state:
    :return: knn movies to past movies
    '''
    df_rec['yearDif'] = df_rec['year'].apply(lambda x: getYearDif(x, yearRef))
    df_rec['yearScore'] = 1 - df_rec['yearDif'] / df_rec['yearDif'].max()
    return df_rec.sort_values(by='yearScore', ascending=False)[:n]

###########################################
#         Load and save past movie        #
###########################################
def getPastMovies():
    l = loadPastMovies()
    if len(l) == 0:
        return None
    return df[df['imdbId'].isin(l)]


def loadPastMovies():
    fileName = 'pastMovies.csv'
    if os.path.isfile(fileName):
        try:
            mov = pd.read_csv(fileName, header=None)
            return mov[0].tolist()
        except Exception as e:
            return []
    return []


def savePastMovies(mov):
    fileName = 'pastMovies.csv'
    mov = pd.DataFrame(mov)
    mov.to_csv(fileName, header=None, index=False)


###########################################
#       Print Selecting Board CLI         #
###########################################
import os
def clear_terminal():
    if 'TERM' in os.environ:
        os.system('clear')
    else:
        print("\n" * 100)
def printSelectingBoard(df):
    print(f'{"=" * 20}{"=" * 20}{"=" * 20}')
    print(f'{"=" * 20}{"Movie Searching": ^20}{"=" * 20}')
    print(f'{"=" * 20}{"=" * 20}{"=" * 20}')

    recMov = getMovieRec(df).reset_index()[['imdbId', 'title', 'genres']]
    print(recMov)
    print()
    print(f'Select a movie by its index to get Recommendation.')
    print(f'Ttype st to search by title.')
    print(f'Type sg to search by genre.')
    print(f'Type stg to search by genre then title.')
    print(f'Type q to quit.')
    print()
    opt = input(f'Please enter your option: ')
    clear_terminal()

    if opt.isdigit():
        val = int(opt)
        if val < 0 or val > 9:
            print(f'Invalid input. Please try again.')
            printSelectingBoard(df)
        else:
            # save imdbId to pastMovies.csv
            imdbId = recMov.loc[val]['imdbId']
            pastMovList = loadPastMovies()
            pastMovList.append(imdbId)
            savePastMovies(pastMovList)
    else:
        invalid = False
        if opt == 'q':
            exit()
        elif opt == 'st':
            print(searchByTitle(df))
            pass
        elif opt == 'sg':
            print(searchByGenre(df))
            pass
        elif opt == 'stg':
            print(searchByGenreThenTitle(df))
            pass
        else:
            invalid = True
            print(f'Invalid input. Please try again.')
            printSelectingBoard(df)
        if not invalid:
            prompt = input(f'Press any key to continue, q to quit: ')
            if prompt == 'q':
                exit()
            else:
                clear_terminal()
                printSelectingBoard(df)


# [81441, 259974, 1194238]

# df = editIMDB(df)
df = extractYear(df)
#getMovieRec(df, 10, 0)

print(df['year'].dtype)