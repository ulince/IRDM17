import createNounFeat as cNF
import pickle
import pandas as pd
from collections import OrderedDict
from datetime import datetime
import time

def generateTokenList(l):
    e = []      # empty array
    s = 0       # start of string
    n = len(l)  # end of string
    c = 0       # counter
    while( n-s >= 2):
        e.append(' '.join(l[s:n]))
        n -= 1
        c += 1
        if n == 2:
            n = len(l)
            s += 1
    l += e
    return (l, len(l)+1)

def featScore(tokens, comStr):
    count = 0

    for t in tokens:
        if t in comStr:
            count += 1
    return count

def calcColScore(stCol, comCol, colHead):
    fD = []
    for idx, tok in enumerate(stCol[0]):
        fullDict = {}
        fScore = 0
        try:
            pT = comCol[idx].lower()
        except:
            pT = 'aaaaaaa'
        tok = tok.split(' ')
        # Check Brand
        #if colHead == "featScore_prodTitle":
        if pT.split(' ')[0] == tok[0]:
            fScore = 1
        tokList = generateTokenList(tok)
        #print(tokList)
        #print("===============================")
        tf = tokList[1]
        score = featScore(tokList[0], pT)
        #print(score)
        #print("+++++++++++++++++++++++++++++++")
        fScore = (fScore + score) / tf
        fullDict[colHead] = round(fScore,3)
        fD.append(fullDict)

    return pd.DataFrame(fD)


#####################################################################################
# d = ['hot', 'water', 'heater', 'gas', 'sadas', 'sadad']

all_pd = pd.read_csv('Data_Kaggle/df_all.csv', encoding="ISO-8859-1")

all_pd.drop(['Unnamed: 0', 'Unnamed: 0_x', 'Unnamed: 0_y', 'Unnamed: 0.1'], axis = 1, inplace = True)

trainCSV = '../data/Data_Kaggle/df_all.csv'
pdDescCSV = 'product_descriptions.csv'
attriCSV = 'attributes.csv'

df_train = pd.read_csv(trainCSV, encoding="ISO-8859-1")

# Splitting preposition
stHeadCols = ['product_title']
stCol = df_train[stHeadCols]

cNF.create_nounFeat(stCol)

with open("../data/pickles/full_st_Tokens.p", 'rb') as file:
    fTok = pickle.load(file)
with open("../data/pickles/pre_st_Tokens.p", 'rb') as file:
    preTok = pickle.load(file)
with open("../data/pickles/post_st_Tokens.p", 'rb') as file:
    postTok = pickle.load(file)
with open("../data/pickles/prodTitle.p", 'rb') as file:
    prodTitle = pickle.load(file)
with open("../data/pickles/prodDesc.p", 'rb') as file:
    prodDesc = pickle.load(file)
with open("../data/pickles/prodAttr.p", 'rb') as file:
    prodAttr = pickle.load(file)
with open("../data/pickles/nounFeat.p", 'rb') as file:
    nounFeat = pickle.load(file)

prt = "featScore_prodTitle"
prd = "featScore_prodDesc"
pra = "featScore_prodAttr"
pnf = "featScore_nounFeat"

proT = prodTitle['product_title']
proD = prodDesc['product_description']
proA = prodAttr['name']
proN = nounFeat[0]

# Real Running Block of Codes
print("After this starts the computation of features")
beginTime = datetime.now().time()

featScore1 = calcColScore(fTok, proT, prt)
featScore2 = calcColScore(fTok, proD, prd)
featScore3 = calcColScore(fTok, proA, pra)
#featScore4 = calcColScore(fTok, nounFeat, pnf)

fTok.columns = ['search_term']

allFeat = [fTok, featScore1, featScore2, featScore3]
allFeat = pd.concat(allFeat, axis = 1)
endTime = datetime.now().time()

pickle.dump(allFeat, open("../data/pickles/weng_feat_all" + ".p", "wb"))
allFeat.to_csv("../data/features/weng_feat_all.csv", index = False)

print("Done!")
print("It began at :", beginTime)
print("It ends at :", endTime)
