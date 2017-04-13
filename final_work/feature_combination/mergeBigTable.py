import pickle
import pandas as pd

pd1 = pd.read_csv('Data_Kaggle/train_spelled.csv', encoding="ISO-8859-1")
pd2 = pd.read_csv('allST_allCat_TFIDF_WF_w2v.csv', encoding="ISO-8859-1")
pd3 = pd.read_csv('kamiAllFeat.csv', encoding="ISO-8859-1")
pd4 = pd.read_csv('UlyAllFeat.csv', encoding="ISO-8859-1")

pd1.drop(['Unnamed: 0', 'id', 'product_uid', 'product_title'], axis = 1, inplace = True)
pd2.drop(['Unnamed: 0', 'search_term', 'prodDesc', 'prodTitle', 'Attributes'], axis = 1, inplace = True)
#pd3.drop('Unnamed: 0', axis = 1, inplace = True)
pd4.drop('search_term', axis = 1, inplace = True)

#pd1.columns = ['search_term', 'full_feat1', 'full_feat2', 'full_feat3', 'full_feat4_5']
#pd2.columns = ['pre_feat1', 'pre_feat2', 'pre_feat3', 'pre_feat4_5']
#pd3.columns = ['post_feat1', 'post_feat2', 'post_feat3', 'post_feat4_5']

all_pd = [pd1, pd2, pd3]
all_pd = pd.concat(all_pd, axis = 1)

all_pd.to_csv('Final_Big_Table_noUly.csv', index = False)

