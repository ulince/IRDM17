import numpy as np
import pandas as pd
import re
from datetime import datetime
import pickle

def filter_tokens(data):
    space = ' '
    stopwords = ''
    stopWordsFile = 'stopWords.txt'

    # Creating search term token vector (column)
    # stTokens = pd.DataFrame(np.zeros((len(data),1)))
    # stTokens = stTokens.astype('str')

    preTok = pd.DataFrame(np.zeros((len(data),1))).astype('object')
    postTok = pd.DataFrame(np.zeros((len(data),1))).astype('object')
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
        temp = re.split(stopwords, data.loc[i][0])
        # print("========================================")
        # print(temp)
        if(len(temp) > 1):
            preTok[0][i] = temp[0]
            postTok[0][i] = ' '.join(temp[1:])
            stop[0][i] = '1'
        else:
            preTok[0][i] = postTok[0][i] = 'aaaaaaaaa'
            stop[0][i] = '0'
        # print(preTok[0][i])
        # print(postTok[0][i])
        # print("========================================")
    print(datetime.now().time())

    pickle.dump(preTok, open("../data/pre_st_Tokens" + ".p", "wb"))
    pickle.dump(postTok, open("../data/post_st_Tokens" + ".p", "wb"))
    pickle.dump(stop, open("../data/stopBool_stopwords" + ".p", "wb"))

    return (preTok, postTok, stop)
