import pandas as pd
import pickle
from collections import *
import math
import numpy as np
import filter_stTokens as fsT
from scipy import spatial
from datetime import datetime

# Loading the dictionaries with the corpus. Doc_freq... are the one with one vote per documents the other is
# the full code
with open("../data/pickles/freq_description.p", 'rb') as file:
    freq_description = pickle.load(file)
with open("../data/pickles/freq_attributes.p", 'rb') as file:
    freq_attributes = pickle.load(file)
with open("../data/pickles/doc_freq_description.p", 'rb') as file:
    doc_freq_description = pickle.load(file)
with open("../data/pickles/doc_freq_attributes.p", 'rb') as file:
    doc_freq_attributes = pickle.load(file)
with open("../data/pickles/pre_st_Tokens.p", 'rb') as file:
    preTokens = pickle.load(file)
with open("../data/pickles/post_st_Tokens.p", 'rb') as file:
    postTokens = pickle.load(file)
with open("../data/pickles/final_embeddings.p", 'rb') as file:
    model = pickle.load(file)
with open("../data/pickles/reverse_dictionary.p", 'rb') as file:
    name_dictionary = pickle.load(file)
with open("../data/pickles/dictionary.p", 'rb') as file:
    name_list = pickle.load(file)


# Opening the data, among other thing to have the number N of documents
df_all = pd.read_csv('../data/Data_Kaggle/df_all.csv', encoding="ISO-8859-1")
product_description = pd.read_csv('../data/Data_Kaggle/product_description_spelled.csv', encoding="ISO-8859-1")
attributes = pd.read_csv('../data/Data_Kaggle/attributes.csv', encoding="ISO-8859-1")

# Initializing N_description and N_attribute
N_description = len(product_description.product_description)
N_attribute = len(attributes.value)

#print("N_desc :", N_description)
#print("N_attr :", N_attribute)
#print("N_d2 :", len(df_all['product_description']))
#print("N_d3 :", len(df_all['name']))

# Initializing full search term and pre-processed and post-processed search terms
print("========================================")
print("Computing the three types of queries")
cFsT = df_all['search_term']
# tmp = fsT.filter_tokens(cFsT)
prodDesc = df_all['product_description']
prodTitle = df_all['product_title']
prodAttr = df_all['name']

# Removing NaN item in Attributes
prodAttr = prodAttr.fillna('aaaaaaaaa')

preTokens = preTokens[0].astype('str')
postTokens = postTokens[0].astype('str')

'''
tmp = fsT.filter_tokens(cFsT)
preTokens = tmp[0]
postTokens = tmp[1]

pickle.dump(preTokens, open("preTokens" + ".p", "wb"))
pickle.dump(postTokens, open("postTokens" + ".p", "wb"))
'''

#print(preTokens)
#print("========================================")
#print(postTokens)
print("Queries computation done")

#####################################################################################
#####OLD FEATURE DO NOT IMPLEMENT ############
def doc_weight_per_word(word, corpus, N, document):
    n_t = corpus[word]
    f_td = Counter(document.split(" "))
    f_td = f_td[word]
    doc_weight = f_td*math.log(N/n_t)
    print("n and N",n_t,N)
    print("log",math.log(N/n_t))
    print(doc_weight)

#####################################################################################
# this function receive a query, a corpus of frequency, the number of document in the corpus and a document
# it then compute the tf_idf score for each word in common between the frequency and the document and return the total
# tf_idf weight

# imporetant this cropus should be done with doc_freq...
#####IMPLEMENT ME PLEASE############
def tf_idf(query, corpus, N, document, print_all = False):
    tot_weight = 0
    for word in query.split(" "):
        if word in document.split(" "):
            n_t = max(corpus[word], 1)
            f_td = Counter(document.split(" "))
            f_td = f_td[word]
            weight = f_td*math.log(N/n_t)
            tot_weight += weight
            if print_all:
                print("-----------------", word, "-----------------")
                print("n and N",n_t,N)
                print("log",math.log(N/n_t))
                print(weight)
        else:
            if print_all:
                print("-----------------", word, "-----------------")
                print(word," is not in, MAY I TAKE A MESSAGE")
    return tot_weight

#####################################################################################
# this function compute a weight of a pair document-query according to the frequency of the words in the corpus, not
# the frequency of document using a particular word. We still use the same kind of logic of the tf-idf

# important, this should be "fed" with the corpus freq_attributes.... NOT doc_freq
#####IMPLEMENT ME PLEASE############
def weighting_full_word(query, corpus, document, print_all=False):
    tot_weight = 0
    for word in query.split(" "):
        if word in document.split(" "):
            n_t = max(corpus[word], 1)
            N = sum(corpus.values())
            f_td = Counter(document.split(" "))
            f_td = f_td[word]
            weight = f_td*math.log(N/n_t)
            tot_weight += weight
            if print_all:
                print("-----------------", word, "-----------------")
                print("n and N",n_t,N)
                print("log",math.log(N/n_t))
                print(weight)
        else:
            if print_all:
                print("-----------------", word, "-----------------")
                print(word," is not in, MAY I TAKE A MESSAGE")
    return tot_weight

#####################################################################################
# Word2vec
def feature_word2vec_avg_proximity(query, document, print_detail=False):
    #first we select the vector space present in the document
    doc_domain = list()
    for word in document.split():
        ind = name_list[word]
        model[ind]
        #doc_domain = np.concatenate(doc_domain,(model[ind]))
        doc_domain.append(model[ind])
    #changing format for the tree
    doc_domain = np.array(doc_domain)
    #computing the proximity tree
    tree = spatial.KDTree(doc_domain)

    #for each word in the query we compute the minimum distance to the doc_domain
    score = 0
    number_word_used = 0
    for word in query.split():
        if word in name_list:
            ind = name_list[word]
            dist_object = tree.query(model[ind],k=1)
            number_word_used += 1
            score += dist_object[0]
            if print_detail:
                print("--------------",word,"--------------")
                print(word,"distance score (lowest=best) is ", dist_object[0])
        else:
            # the word is not present in the full word2vec space. We do not penalize it nor count it for the final
            # division
            if print_detail:
                print("--------------",word,"--------------")
                print(word,"is not used because not present in the full word2vec space")
    if number_word_used == 0:
        final_score = 10
    else:
        final_score = score/number_word_used
    if print_detail:
        print("--------------","Final score","--------------")
        print("Brute score:", score)
        print("Number of word in the space:",number_word_used)
        print("Final score:", final_score)
    return final_score

#####################################################################################
# ! Big Table Building Zone
def bigTableGenerator(full_ST, pre_ST, post_ST, prodDescFreq, N_desc, prodAttrFreq, N_attr, prodDesc, prodTitle, prodAttr):
    for i in range(len(full_ST)):
        print(i)
       
        # Computing tf-idf features
        tf_x1 = tf_idf(full_ST.loc[i], prodDescFreq, N_desc, prodDesc.loc[i], False)
        tf_y1 = tf_idf(pre_ST.loc[i], prodDescFreq, N_desc, prodDesc.loc[i], False)
        tf_z1 = tf_idf(post_ST.loc[i], prodDescFreq, N_desc, prodDesc.loc[i], False)
        tf_x2 = tf_idf(full_ST.loc[i], prodDescFreq, N_desc, prodTitle.loc[i], False)
        tf_y2 = tf_idf(pre_ST.loc[i], prodDescFreq, N_desc, prodTitle.loc[i], False)
        tf_z2 = tf_idf(post_ST.loc[i], prodDescFreq, N_desc, prodTitle.loc[i], False)
        tf_x3 = tf_idf(full_ST.loc[i], prodAttrFreq, N_attr, prodAttr.loc[i], False)
        tf_y3 = tf_idf(pre_ST.loc[i], prodAttrFreq, N_attr, prodAttr.loc[i], False)
        tf_z3 = tf_idf(post_ST.loc[i], prodAttrFreq, N_attr, prodAttr.loc[i], False)
  
        # Computing weighthing_full_word features
        wf_x1 = weighting_full_word(full_ST.loc[i], prodDescFreq, prodDesc.loc[i], False)
        wf_y1 = weighting_full_word(pre_ST.loc[i], prodDescFreq, prodDesc.loc[i], False)
        wf_z1 = weighting_full_word(post_ST.loc[i], prodDescFreq, prodDesc.loc[i], False)
        wf_x2 = weighting_full_word(full_ST.loc[i], prodDescFreq, prodTitle.loc[i], False)
        wf_y2 = weighting_full_word(pre_ST.loc[i], prodDescFreq, prodTitle.loc[i], False)
        wf_z2 = weighting_full_word(post_ST.loc[i], prodDescFreq, prodTitle.loc[i], False)        
        wf_x3 = weighting_full_word(full_ST.loc[i], prodAttrFreq, prodAttr.loc[i], False)
        wf_y3 = weighting_full_word(pre_ST.loc[i], prodAttrFreq, prodAttr.loc[i], False)
        wf_z3 = weighting_full_word(post_ST.loc[i], prodAttrFreq, prodAttr.loc[i], False)

        # Computing word2vec_avg_proximity
        wv_x1 = feature_word2vec_avg_proximity(full_ST.loc[i], prodDesc.loc[i], False)
        wv_y1 = feature_word2vec_avg_proximity(pre_ST.loc[i], prodDesc.loc[i], False)
        wv_z1 = feature_word2vec_avg_proximity(post_ST.loc[i], prodDesc.loc[i], False)
        '''
        wv_x2 = feature_word2vec_avg_proximity(full_ST.loc[i], prodTitle.loc[i], False)
        wv_y2 = feature_word2vec_avg_proximity(pre_ST.loc[i], prodTitle.loc[i], False)
        wv_z2 = feature_word2vec_avg_proximity(post_ST.loc[i], prodTitle.loc[i], False)
        wv_x3 = feature_word2vec_avg_proximity(full_ST.loc[i], prodAttr.loc[i], False)
        wv_y3 = feature_word2vec_avg_proximity(pre_ST.loc[i], prodAttr.loc[i], False)
        wv_z3 = feature_word2vec_avg_proximity(post_ST.loc[i], prodAttr.loc[i], False)       
        '''

        yield [full_ST.loc[i], prodDesc.loc[i], prodTitle.loc[i], prodAttr.loc[i], tf_x1, tf_y1, tf_z1, tf_x2, tf_y2, tf_z2, tf_x3, tf_y3, tf_z3, wf_x1, wf_y1, wf_z1, wf_x2, wf_y2, wf_z2, wf_x3, wf_y3, wf_z3, wv_x1, wv_y1, wv_z1]      

#####################################################################################
# Real Running Block of Codes
print("After this starts the computation of features")
beginTime = datetime.now().time()

#Uncomment for full 3 categories
# bigTable = pd.DataFrame(columns=('search_term', 'prodDesc', 'prodTitle', 'Attributes', 'FST_TFPD_Score', 'preST_TFPD_Score', 'postST_TFPD_Score', 'FST_TFPT_Score', 'preST_TFPT_Score', 'postST_TFPT_Score', 'FST_TFAT_Score', 'preST_TFAT_Score', 'postST_TFAT_Score'))

# Current Table
bigTable = pd.DataFrame(columns=('search_term', 'prodDesc', 'prodTitle', 'Attributes', 'FST_TFPD_Score', 'preST_TFPD_Score', 'postST_TFPD_Score', 'FST_TFPT_Score', 'preST_TFPT_Score', 'postST_TFPT_Score', 'FST_TFAT_Score', 'preST_TFAT_Score', 'postST_TFAT_Score', 'FST_WFPD_Score', 'preST_WFPD_Score', 'postST_WFPD_Score', 'FST_WFPT_Score', 'preST_WFPT_Score', 'postST_WFPT_Score', 'FST_WFAT_Score', 'preST_WFAT_Score', 'postST_WFAT_Score', 'FST_W2VPD_Score', 'preST_W2VPD_Score', 'postST_W2VPD_Score'))

bTGen = bigTableGenerator(cFsT, preTokens, postTokens, doc_freq_description, N_description, doc_freq_attributes, N_attribute, prodDesc, prodTitle, prodAttr)

j = 0

for k in bTGen:
    bigTable.loc[j] = k
    j += 1

pickle.dump(bigTable, open("../data/pickles/allST_allCat_TFIDF_WF_w2v" + ".p", "wb"))

endTime = datetime.now().time()

print("Table as shown : ")
print(bigTable)

bigTable.to_csv("../data/features/allST_allCat_TFIDF_WF_w2v.csv")

print("Done!")
print("It began at :", beginTime)
print("It ends at :", endTime)

