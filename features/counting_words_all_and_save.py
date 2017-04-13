import pandas as pd
import pickle
from collections import Counter

df_train = pd.read_csv('Data_Kaggle/train.csv', encoding="ISO-8859-1")
product_description = pd.read_csv('Data_Kaggle/product_description_spelled.csv', encoding="ISO-8859-1")
attributes = pd.read_csv('Data_Kaggle/attributes.csv', encoding="ISO-8859-1")
df_all = pd.merge(df_train, product_description, how='left', on='product_uid')
df_all = pd.merge(df_all, attributes, how='left', on='product_uid')


all_description = ""
all_attributes = ""

doc_freq_description = Counter("start".split())
doc_freq_attributes = Counter("start".split())
freq_description = Counter("start".split())
freq_attributes = Counter("start".split())


def merge_counter(a, b):
    fc = a+b
    return fc

for i in range(len(product_description.product_description)):
    des = product_description.product_description[i]
    if type(des) is str:
        doc_freq_description = merge_counter(doc_freq_description, Counter(set(des.split(" "))))
        freq_description = merge_counter(freq_description, Counter(des.split(" ")))
    if i % 100 == 10:
        print("Description frequency processing: ", round(100*i/len(product_description.product_description),2),"%")

pickle.dump(freq_description, open("freq_description" + ".p", "wb"))
pickle.dump(doc_freq_description, open("doc_freq_description" + ".p", "wb"))

for i in range(len(attributes.value)):
    att = attributes.value[i]
    if type(att) is str:
        doc_freq_attributes = merge_counter(doc_freq_attributes, Counter(set(att.split(" "))))
        freq_attributes = merge_counter(freq_attributes, Counter(att.split(" ")))

    if i % 100 == 10:
        print("Attributes frequency processing: ", round(100*i/len(attributes.value),2),"%")


pickle.dump(freq_attributes, open("freq_attributes" + ".p", "wb"))
pickle.dump(doc_freq_attributes, open("doc_freq_attributes" + ".p", "wb"))



