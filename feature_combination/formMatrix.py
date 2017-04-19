import pandas as pd
import csv
import ast
from sklearn.feature_extraction import DictVectorizer
import pickle
from collections import defaultdict

vectorizer = DictVectorizer()
bT_noUly = "../data/features/allFeat_noUly.csv"
bigPD = pd.read_csv(bT_noUly, encoding="ISO-8859-1")


big_table_path = "../data/features/Final_Big_Table_noUly.csv"
target_table_path = "../data/features/weng_feat_all.csv"

pd1 = pd.read_csv(big_table_path, encoding="ISO-8859-1")
pd2 = pd.read_csv(target_table_path, encoding="ISO-8859-1")

tpd = pd2.drop('search_term', axis = 1)
fpd = [pd1, tpd]
fpd = pd.concat(fpd, axis = 1)
fpd.to_csv("../data/allFeat_noUly.csv", index = False)
print(list(fpd))


def pickle_object(obj, path):
    pickle.dump(obj,open(path,"wb"))

#Loads the data from Final_Big_Table and returns:
#features is a list of dictionaries, with each dictionary containing the feature name-value pairs for the samples
#labels is a list containing the relevance scores for each corresponding element in the features dictionary.
#search_terms is a list containing the search term for each corresponding element in the features dictionary.
def load_data(filepath:str):
    features = []
    labels = []
    search_terms = []
    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            instance = {'FST_TFPD_Score':float(row[2]),
                       'preST_TFPD_Score':float(row[3]),
                       'postST_TFPD_Score':float(row[4]),
                       'FST_TFPT_Score':float(row[5]),
                       'preST_TFPT_Score':float(row[6]),
                       'postST_TFPT_Score':float(row[7]),
                       'FST_TFAT_Score':float(row[8]),
                       'preST_TFAT_Score':float(row[9]),
                       'postST_TFAT_Score':float(row[10]),
                       'FST_WFPD_Score':float(row[11]),
                       'preST_WFPD_Score':float(row[12]),
                       'postST_WFPD_Score':float(row[13]),
                       'FST_WFPT_Score':float(row[14]),
                       'preST_WFPT_Score':float(row[15]),
                       'postST_WFPT_Score':float(row[16]),
                       'FST_WFAT_Score':float(row[17]),
                       'preST_WFAT_Score':float(row[18]),
                       'postST_WFAT_Score':float(row[19]),
                       'FST_W2VPD_Score':float(row[20]),
                       'preST_W2VPD_Score':float(row[21]),
                       'postST_W2VPD_Score':float(row[22]),
                       'prodTitle_FsT_prob':float(row[23]),
                       'prodTitle_length_feat':float(row[24]),
                       'prodTitle_postST_prob':float(row[25]),
                       'prodTitle_preST_prob':float(row[26]),
                       'prodDesc_FsT_prob':float(row[27]),
                       'prodDesc_length_feat':float(row[28]),
                       'prodDesc_postST_prob':float(row[29]),
                       'prodDesc_preST_prob':float(row[30]),
                       'prodAttr_FsT_prob':float(row[31]),
                       'prodAttr_length_feat':float(row[32]),
                       'prodAttr_postST_prob':float(row[33]),
                       'prodAttr_preST_prob':float(row[34]),
                       'featScore_prodTitle':float(row[35]),
                       'featScore_prodDesc':float(row[36]),
                       'featScore_prodAttr':float(row[37])
                       #'featScore_nounFeat':float(row[38])
                        }
            #full_feat1 = ast.literal_eval(row[35])
            #if full_feat1['subsequence_query+title']:
            #    instance.update({'full_feat1':full_feat1['subsequence_query+title']})
            #full_feat2 = ast.literal_eval(row[36])
            #if full_feat2['subsequence_query+description']:
            #    instance.update({'full_feat2':full_feat2['subsequence_query+description']})
            #instance.update(ast.literal_eval(row[37]))
            #instance.update(ast.literal_eval(row[38]))
            #temp = ast.literal_eval(row[38])
            #instance.update({'common_word_count':int(temp['common_word_count'])})


            features.append(instance)
            search_terms.append(row[0])
            labels.append(float(row[1]))
    return features, search_terms,labels

#############################################################################################
features,search_terms,labels = load_data(bT_noUly)

matrix_form = vectorizer.fit_transform(features)

pickle.dump(matrix_form, open("../data/pickles/allfeatNoUlyMat" + ".p", "wb"))
