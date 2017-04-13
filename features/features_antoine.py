import pandas as pd
import pickle
from collections import *
import math

# loading the dictionaries with the corpus. Doc_freq... are the one with one vote per documents the other is
# the full code
with open("freq_description.p", 'rb') as file:
    freq_description = pickle.load(file)
with open("freq_attributes.p", 'rb') as file:
    freq_attributes = pickle.load(file)
with open("doc_freq_description.p", 'rb') as file:
    doc_freq_description = pickle.load(file)
with open("doc_freq_attributes.p", 'rb') as file:
    doc_freq_attributes = pickle.load(file)
print(doc_freq_description)
# opening the data, among other thing to have the number N of documents
df_train = pd.read_csv('Data_Kaggle/train.csv', encoding="ISO-8859-1")
product_description = pd.read_csv('Data_Kaggle/product_description_spelled.csv', encoding="ISO-8859-1")
attributes = pd.read_csv('Data_Kaggle/attributes.csv', encoding="ISO-8859-1")
df_all = pd.read_csv('Data_Kaggle/df_all.csv', encoding="ISO-8859-1")

print(df_all.head())
N_description = len(product_description.product_description)
N_attribute = len(attributes.value)

#####OLD FEATURE DO NOT IMPLEMENT ############
def doc_weight_per_word(word, corpus, N, document):
    n_t = corpus[word]
    f_td = Counter(document.split(" "))
    f_td = f_td[word]
    doc_weight = f_td*math.log(N/n_t)
    print("n and N",n_t,N)
    print("log",math.log(N/n_t))
    print(doc_weight)


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


def test_loop():
    for i in range(300):
        print("-------------------------------------------------------------------",i,
              "-------------------------------------------------------------------")
        att = df_all.name[i]
        quer = df_all.search_term[i]

        a =tf_idf(quer,doc_freq_attributes,N_attribute,att, True)
        print("FINALY:", a)

i = 198
att = df_all.name[i]
quer = df_all.search_term[i]
print("attribute", att)
print("query", quer)