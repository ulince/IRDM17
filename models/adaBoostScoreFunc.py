import pandas as pd
import pickle
import random
from collections import Counter
import numpy as np
import math
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPRegressor
import time
import warnings
import random

random.seed(42)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


################################## Functions #######################################################
# input: an ordered vector of relevance, output: Discountegd Cumulative Gain
def DCG(vec):
    sc = 0
    for i in range(1,len(vec)):
        sc += ((2**vec[i-1])-1)/math.log(i+1, 2)
    return sc

################################## Loading dat and matrix###########################################

#targetPickle = "pickles/allfeatNoUlyMat.p"
targetPickle = "pickles/the_matrix_0feat.p"

with open(targetPickle, 'rb') as file:
    features_mat = pickle.load(file)

df_all = pd.read_csv('Data_Kaggle/df_all.csv', encoding="ISO-8859-1")

################################## Main code #######################################################

unique_search_term = list(set(df_all.search_term))
# K fold cross validation
K = 3

#tree_size = [50,100,300]
#tree_size = [1,5,10,50,100,300]
tree_size = [1,5,10,50,100,300,600]
#tree_size = [1,5,10,50,100,300,600, 800, 1000, 1200, 1500]
mean_score_DCG = np.empty(len(tree_size))
mean_score_RMSE= np.empty(len(tree_size))
for t in range(len(tree_size)):
    ts = tree_size[t]
    print("//////////////////////", "Starting with tree size =", ts, "//////////////////////")
    scores_in_cross_DCG = np.empty(K)
    scores_in_cross_RMSE = np.empty(K)
    # percentage of the training set set asside for testing in k cross validation
    test_percentage = 0.4
    for k in range(K):
        print("--------------- Starting round:", k+1, "/", K, "---------------")
        print("Spliting data randomly")
        # we select some random query for test set
        test_queries = random.sample(unique_search_term,round(test_percentage*len(unique_search_term)))
        train_queries = list(set(unique_search_term)-set(test_queries))
        # we select the sub data frame with only the train and test_queries
        df_train = df_all.loc[df_all['search_term'].isin(train_queries)].copy()
        df_test = df_all.loc[df_all['search_term'].isin(test_queries)].copy()

        # we then get the indexes for the feature matrix split
        ind_train = df_all[df_all['search_term'].isin(train_queries)].index.tolist()
        ind_test = df_all[df_all['search_term'].isin(test_queries)].index.tolist()
        # splitting the feature matrix into train and test set
        features_train = features_mat.tocsc()[ind_train,:]
        #features_train = features_train.toarray()
        features_test = features_mat.tocsc()[ind_test,:]
        #features_test = features_test.toarray()

        ####################### train the model on df_train #####################################
        print("Training model")
        start = time.time()
        # TODO train your model here with features_train as an input
        Y = df_train.relevance
        model = AdaBoostRegressor(n_estimators = ts)
        #model = GradientBoostingRegressor(n_estimators = ts, max_depth = 10)
        #model = DecisionTreeRegressor(max_depth = 10)
        #model = ExtraTreesRegressor(n_estimators = ts, max_depth = 10)
        #model = AdaBoostRegressor(RandomForestRegressor(n_estimators = ts, max_depth = 10), n_estimators = 5)
        #model = RandomForestRegressor(n_estimators = ts, max_depth=10)
        #model = MLPRegressor()
        #model = LogisticRegression()
        model = model.fit(y=Y, X= features_train)
        end = time.time()
        print("Model trained in",end - start)
        ####################### testing #########################################################
        # TODO apply your model to the df_test and put the predicted relevance in the "est_relevance" column
        estimated_relevance = model.predict(features_test)
        #estimated_relevance  = np.random.randint(3, size=(len(df_test.search_term)))
        df_test.loc[:,"est_relevance"] = estimated_relevance # random solution, np.random.randint(3, size=(len(df_test.search_term)))
        y = np.array(df_test.relevance)
        y_hat = np.array(estimated_relevance)

        # Computing the final score
        # First we create a list of documents organized by our model
        final_k_score_DCG = 0
        c = 0
        #computing the dcg score
        for query in test_queries:
            df_temp = df_test.loc[df_test.search_term == query]
            df_temp = df_temp.sort(['est_relevance'],ascending = False)
            if len(df_temp.relevance)>1:
                c += 1
                # applying the normalized DCG by computing it and divided it by the perfect DCG score
                final_k_score_DCG += DCG(np.array(df_temp.relevance))/DCG(-np.sort(-np.array(df_temp.relevance)))
                #print(DCG(np.array(df_temp.relevance))/DCG(-np.sort(-np.array(df_temp.relevance))))
        # divding the usm of normaized DCG score by the number of queries with more than 1 document
        final_k_score_DCG = final_k_score_DCG/c
        # computing the root means square error

        final_k_score_RMSE = np.power(np.sum(np.power(y - y_hat, 2)) / len(y), 0.5)

        print( "--------","DCG, score of the",k+1,"step is:",final_k_score_DCG, "-----")
        print( "--------","RMSE, score of the",k+1,"step is:",final_k_score_RMSE, "-----")

        #saving final score
        scores_in_cross_DCG[k] = final_k_score_DCG
        scores_in_cross_RMSE[k] = final_k_score_RMSE

    print("***Cross validation procedure completed***")
    print("DCG score vector", scores_in_cross_DCG)
    print("DCG mean score", np.mean(scores_in_cross_DCG))
    print("RMSE score vector", scores_in_cross_RMSE)
    print("RMSE mean score", np.mean(scores_in_cross_RMSE))
    mean_score_DCG[t] = np.mean(scores_in_cross_DCG)
    mean_score_RMSE[t] = np.mean(scores_in_cross_RMSE)
print("****************", "full procedure completed","****************")
print("Mean score dcg:")
print(mean_score_DCG)
print("Mean score RMSE:")
print(mean_score_RMSE)

final = list()
final.append(mean_score_RMSE)
final.append(mean_score_DCG)
