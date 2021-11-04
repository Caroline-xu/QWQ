import pandas as pd
from feature_lb import feature_lb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

from train_linear_regression import train_linear_regression
# train_linear_regression() is defined in linear_regression.py
#from linear_regression import train_linear_regression
#from decision_tree import train_decision_tree
#from naive_bayes import train_naive_bayes
#from random_forest import train_random_forest
def get_data():
    # Loads the features dataset to a pandas dataframe and returns it
# Load the Diabetes dataset
    df = feature_lb()
    #print(df)
    return df

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
    
    #split data into 0.8 train, 0.2 test
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    
    # 2- Cross-fold validation on the train data

   
    '''kf = KFold(n_splits=5) # Define the split - into 2 folds 
    kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
    print(kf) 
    KFold(n_splits=5, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]'''
    '''kf = KFold(n_splits=5) # Define the split - into 5 folds 
    kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
    print(kf) 
    KFold(n_splits=5, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        res = train_linear_regression(X_train, X_test, y_train, y_test) 
        sum += res
    average_sum = sum/5
    print("average accuracy:", average_sum)'''
              
    
# 3- Train a model
   #by changing this model name to get different model
    model_name = 'linear_regression'
    res = train_linear_regression(X_train, X_test, y_train, y_test)
    print("linear regression's accuracy:", res) 
    
    
    '''if model_name == 'linear_regression':
       res = train_linear_regression(X_train, X_test, y_train, y_test)
    elif model_name == 'decision_tree':
        res = train_decision_tree()
    elif model_name == 'naive_bayes':
        res = train_naive_bayes()
    elif model_name == 'random_forest':
        res = train_random_forest()
      
    print("linear regression's result:", res)
    # 4- print the results'''