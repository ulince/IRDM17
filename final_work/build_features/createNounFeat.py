import numpy as np
import pandas as pd
import re
from datetime import datetime
import pickle
import nltk
import string


def filter_tokens(data):
    space = ' '
    stopwords = ''
    stopWordsFile = '../data/stopWords.txt'
    
    # Creating search term token vector (column)
    # stTokens = pd.DataFrame(np.zeros((len(data),1)))
    # stTokens = stTokens.astype('str')
    
    fullTok = pd.DataFrame(np.zeros((len(data),1))).astype('object')
    preTok = pd.DataFrame(np.zeros((len(data),1))).astype('object')
    postTok = pd.DataFrame(np.zeros((len(data),1))).astype('object')
    filterTok = pd.DataFrame(np.zeros((len(data),1))).astype('object')
    stop = pd.DataFrame(np.zeros((len(data),1))).astype('str')
    
    print(datetime.now().time())
    
    # Read file with stop words
    with open(stopWordsFile) as f:
        stopWList = f.readlines()
    
    for i in range(len(stopWList)):
        stopWList[i] = space + space.join(stopWList[i].split('\n'))
        stopwords += stopWList[i] + '|'
    
    stopwords = stopwords[:-1]

    # Inserting processed search term column into search term token column
    for i in range(len(data)):
    
        fullTemp = re.sub('xbi', 'x', data.loc[i][0])
        fullTemp = ' '.join(re.findall('(\d+|\D+)', fullTemp))
        fullTok[0][i] = fullTemp
        temp = re.split(stopwords, fullTemp)
        #print("========================================")
        #print(temp)
        if(len(temp) > 1):
            preTok[0][i] = temp[0]
            postTok[0][i] = ' '.join(temp[1:])
            stop[0][i] = '1'
    else:
        preTok[0][i] = postTok[0][i] = 'aaaaaaaaa'
            stop[0][i] = '0'

    print(datetime.now().time())
    
    df_tr = pd.read_csv('../data/Data_Kaggle/train_spelled.csv', encoding="ISO-8859-1")
    df_tr.drop('Unnamed: 0', axis = 1, inplace = True)
    
    #pTok1 = preTok
    #pTok1.columns = ['search_term']
    #preTokTrain = [pTok1, df_tr]
    #preTokTrain = pd.concat(preTokTrain, axis = 1)
    
    fullTokTrain = df_tr
    fullTokTrain['search_term'] = fullTok
    fullTokTrain.to_csv('../data/Data_Kaggle/fullTokTrain.csv', index = False)
    
    preTokTrain = df_tr
    preTokTrain['search_term'] = preTok
    preTokTrain.to_csv('../data/Data_Kaggle/preTokTrain.csv', index = False)
    
    #pTok2 = postTok
    #pTok2.columns = ['search_term']
    #postTokTrain = [pTok2, df_tr]
    #postTokTrain = pd.concat(postTokTrain, axis = 1)
    postTokTrain = df_tr
    postTokTrain['search_term'] = postTok
    postTokTrain.to_csv('../data/Data_Kaggle/postTokTrain.csv', index = False)
    
    pickle.dump(fullTok, open("../data/pickles/full_st_Tokens" + ".p", "wb"))
    pickle.dump(preTok, open("../data/pickles/pre_st_Tokens" + ".p", "wb"))
    pickle.dump(postTok, open("../data/pickles/post_st_Tokens" + ".p", "wb"))
    pickle.dump(stop, open("../data/pickles/stopBool_stopwords" + ".p", "wb"))
    
    return (preTok, postTok, stop)



def create_nounFeat(data):
    space = ' '
    stopwords = ''
    stopWordsFile = '../data/stopWords.txt'

    # Creating search term token vector (column)
    # stTokens = pd.DataFrame(np.zeros((len(data),1)))
    # stTokens = stTokens.astype('str')

    nounFeat = pd.DataFrame(np.zeros((len(data),1))).astype('object')

    print(datetime.now().time())

    # Read file with stop words
    with open(stopWordsFile) as f:
        stopWList = f.readlines()

    for i in range(len(stopWList)):
        stopWList[i] = space + space.join(stopWList[i].split('\n'))
        stopwords += stopWList[i] + '|'

    stopwords = stopwords[:-1]

    # Inserting processed search term column into search term token column
    translator = str.maketrans('', '', string.punctuation)

    for i in range(len(data)):
        temp = re.split(stopwords, data.loc[i][0].lower())
        temp = list(filter(None, (temp[0].translate(translator).split(' '))))
        #print("========================================")
        #print(temp)
        res = [elem[0] for elem in nltk.pos_tag(temp) if 'NN' in elem[1]]
        if len(res) >= 3:
            nounFeat[0][i] = ' '.join(res[-3:])
        else:
            nounFeat[0][i] = ' '.join(res)

        #print("========================================")
        #print(temp)

    print(datetime.now().time())

    pickle.dump(nounFeat, open("../data/pickles/nounFeat" + ".p", "wb"))


# a = 'Rheem Performance Plus 50 Gal. Tall 9 Year 40,000 BTU High Efficiency Natural Gas Water Heater     '
# a = list(filter(None, a.split(' ')))
# [i for i in nltk.pos_tag(a.split(' ')) if 'NN' in i[1]]
