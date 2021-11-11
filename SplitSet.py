import pandas as pd
from feature_lb import feature_lb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
# from train_linear_regression import train_linear_regression
# train_linear_regression() is defined in linear_regression.py
#from linear_regression import train_linear_regression
#from decision_tree import train_decision_tree
#from naive_bayes import train_naive_bayes
#from random_forest import train_random_forest
from random_forest import train_random_forest
# for standardize data
from sklearn.preprocessing import StandardScaler
# import PCA
from sklearn.decomposition import PCA

# call this function to normalize the data (scalling) and use PCA to reduce dimension(features) to 300 
# take X as features and Y as label (unnormailzied)
def PCA_reduce_dimension(X, Y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=300)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents)
    finalDf = pd.concat([principalDf, Y], axis = 1)
    #print(finalDf)
    return finalDf

def get_data():
    df = feature_lb()
    X = df.loc[:, df.columns != 'New Label']
    y = df['New Label']
    final_df = PCA_reduce_dimension(X,y)
    #print(df)
    return final_df

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # get the features
    #df = get_data()
    # 1- Split data to train and test
    df = get_data()
    #X is whole dataframe except label column 
    X = df.loc[:, df.columns != 'New Label']
    #y is label column
    y = df['New Label']

    """
    #####train model for only once###
    #split data into 0.8 train, 0.2 test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model_name = 'linear_regression'
    res = train_linear_regression(X_train, X_test, y_train, y_test)
    print("linear regression's accuracy:", res) 
    """
    
    ####train model by using 5 fold cross validation####
    # 2- Cross-fold validation on the train data
    k = 5
    kf = KFold(n_splits=k, random_state=None)
    acc_score = []


    # random forest calssification 
    for train_index , test_index in kf.split(X):
        X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]
        train_random_forest(X_train, X_test, y_train, y_test)
        


''' Linear Regression model
    for train_index , test_index in kf.split(X):
        X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]
        acc = train_linear_regression(X_train, X_test, y_train, y_test)
        acc_score.append(acc)
    avg_acc_score = sum(acc_score)/k
    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))
'''



# 3- Train a model
   #by changing this model name to get different model

    
'''if model_name == 'linear_regression':
       res = train_linear_regression(X_train, X_test, y_train, y_test)
    elif model_name == 'decision_tree':
        res = train_decision_tree()
    elif model_name == 'naive_bayes':
        res = train_naive_bayes()
    elif model_name == 'random_forest':
        res = train_random_forest()'''
      
