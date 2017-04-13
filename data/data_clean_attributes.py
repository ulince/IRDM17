import numpy as np
import re
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from collections import Counter

space = ' '
stopwords = ''
trainCSV = 'train.csv'
pdDescCSV = 'product_descriptions.csv'
attriCSV = 'attributes.csv'
stopWordsFile = 'stopWords.txt'

# Splitting preposition
stHeadCols = ['search_term']
df_train.to_csv("search_term_ori.csv",columns=stHeadCols)
stCol = df_train[stHeadCols]

# Creating search term token vector (column)
stTokens = pd.DataFrame(np.zeros((len(stCol),1)))
stTokens = stTokens.astype('object')

# Read file with stop words
with open(stopWordsFile) as f:
    stopWList = f.readlines()

for i in range(len(stopWList)):
    stopWList[i] = space + space.join(stopWList[i].split('\n'))
    stopwords += stopWList[i] + '|'

stopwords = stopwords[:-1]
print(stopwords)

tokenDict = Counter()

# Inserting processed search term column into search term token column
for i in range(len(stCol)):
    temp = re.split(stopwords, stCol.loc[i][0])
    for token in temp:
        tokenDict[token] +=1
    stTokens[0][i] = temp

#print(tokenDict)

stTokens.to_csv("search_term_split.csv")
print(stTokens)
