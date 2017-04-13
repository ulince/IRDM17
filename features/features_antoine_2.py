import pickle
from scipy import spatial
import pandas as pd
import numpy as np

with open("final_embeddings.p", 'rb') as file:
    model = pickle.load(file)
with open("reverse_dictionary.p", 'rb') as file:
    name_dictionary = pickle.load(file)
with open("dictionary.p", 'rb') as file:
    name_list = pickle.load(file)

df_all = pd.read_csv('Data_Kaggle/df_all.csv', encoding="ISO-8859-1")


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


for i in range(len(df_all.search_term)):
    print("---------------------------------------", i,"---------------------------------------")
    quer = df_all.search_term[i]
    doc = df_all.product_description[i]
    feature_word2vec_avg_proximity(query=quer, document=doc)