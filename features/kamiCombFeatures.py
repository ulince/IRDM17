import pandas as pd
import pickle
import math
import numpy as np
#import filter_stTokens as fsT

from collections import *
from scipy import spatial
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

with open("../data/pickles/pre_st_Tokens.p", 'rb') as file:
    preTokens = pickle.load(file)
with open("../data/pickles/post_st_Tokens.p", 'rb') as file:
    postTokens = pickle.load(file)

# Opening the data, among other thing to have the number N of documents
df_all = pd.read_csv('../data/Data_Kaggle/df_all.csv', encoding="ISO-8859-1")
product_description = pd.read_csv('../data/Data_Kaggle/product_description_spelled.csv', encoding="ISO-8859-1")
attributes = pd.read_csv('../data/Data_Kaggle/attributes.csv', encoding="ISO-8859-1")

# Initializing N_description and N_attribute
N_description = len(product_description.product_description)
N_attribute = len(attributes.value)

# Initializing full search term and pre-processed and post-processed search terms
print("========================================")
print("Computing the three types of queries")
cFsT = df_all['search_term']
prodDesc = df_all['product_description']
prodTitle = df_all['product_title']
prodAttr = df_all['name']  #attributes
relScore = df_all['relevance']

# Removing NaN item in Attributes
prodAttr = prodAttr.fillna('aaaaaaaaa')

# Compute pre and post tokens
preTokens = preTokens[0].astype('str')
postTokens = postTokens[0].astype('str')

spPreTokens = preTokens.str.split()
preTokList = spPreTokens.values.tolist()

spPostTokens = postTokens.str.split()
postTokList = spPostTokens.values.tolist()

# Split full search term into tokens
spcFsT = cFsT.str.split()
sTList = spcFsT.values.tolist()

print("Queries computation done")

#####################################################################################
def calcSTProb(dataCol, search_term, prodFeat):
    all_features_list=[]

    for idx, row_data in enumerate(dataCol):
        count = 0
        dict = {}
        for word in search_term[idx]:
            if (row_data.lower().find(word)!= -1):
                count += 1
            feature= count / len(search_term[idx])
        dict["search_term_words_probability"]= round(feature,2)
        dict["Product_length_feature"]= len(prodFeat[idx])
        all_features_list.append(dict)  

    return pd.DataFrame(all_features_list)

#####################################################################################
def calcAllSTProb(prodFeat, featName, FsT, preST, postST):
    all_features_list=[]

    for idx, row_data in enumerate(prodFeat):
        count1 = 0
        count2 = 0
        count3 = 0
        dict = {}

        for word1 in postST[idx]:
            if (row_data.lower().find(word1)!= -1):
                count1 += 1
            feature1 = count1 / len(postST[idx])

        for word2 in preST[idx]:
            if (row_data.lower().find(word2)!= -1):
                count2 += 1
            feature2 = count2 / len(preST[idx])

        for word3 in FsT[idx]:
            if (row_data.lower().find(word3)!= -1):
                count3 += 1
            feature3 = count3 / len(FsT[idx])

        dict[featName + '_postST_prob']= round(feature1, 2)
        dict[featName + '_preST_prob']= round(feature2, 2)
        dict[featName + '_FsT_prob']= round(feature3, 2)
        dict[featName + '_length_feat']= len(prodFeat[idx])
        all_features_list.append(dict)

    return pd.DataFrame(all_features_list)

#####################################################################################
# Real Running Block of Codes
print("After this starts the computation of features")
beginTime = datetime.now().time()

PTList = calcAllSTProb(prodTitle, 'prodTitle', sTList, preTokList, postTokList)
PDList = calcAllSTProb(prodDesc, 'prodDesc', sTList, preTokList, postTokList)
PAList = calcAllSTProb(prodAttr, 'prodAttr', sTList, preTokList, postTokList)

endTime = datetime.now().time()

print(PTList[1:10])

allFeat = [PTList, PDList, PAList]
allFeat = pd.concat(allFeat, axis=1)

print("=====================================================")
pickle.dump(allFeat, open("../data/pickles/kamiAllFeat" + ".p", "wb"))
allFeat.to_csv("../data/features/kamiAllFeat.csv", index = False)
print(allFeat)

print("Done!")
print("It began at :", beginTime)
print("It ends at :", endTime)

