import pandas as pd
import pickle
from collections import Counter

df_train = pd.read_csv('Data_Kaggle/train_spelled.csv', encoding="ISO-8859-1")
product_description = pd.read_csv('Data_Kaggle/product_description_spelled.csv', encoding="ISO-8859-1")
attributes = pd.read_csv('Data_Kaggle/attributes.csv', encoding="ISO-8859-1")
df_all = pd.merge(df_train, product_description, how='left', on='product_uid')
print("all",df_all.shape)
attributes = attributes.groupby(["product_uid"]).agg(lambda x: ' '.join(set(x)))
print(attributes.head())
print("type",type(attributes))
print("test")
print("attribute",attributes.shape)


attributes = pd.read_csv('Data_Kaggle/attributes_spelled.csv', encoding="ISO-8859-1")
#attributes.to_csv('Data_Kaggle/attributes_two.csv')
df_all = pd.merge(df_all, attributes, how='left', on='product_uid')
print("Train",df_train.shape)
print("all",df_all.shape)

df_all.to_csv('Data_Kaggle/df_all.csv')

