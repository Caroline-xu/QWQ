import pandas as pd
from linearRegression import linear_regression
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
# train_linear_regression() is defined in linear_regression.py
#from linear_regression import train_linear_regression
#from decision_tree import train_decision_tree
#from naive_bayes import train_naive_bayes
#from random_forest import train_random_forest
def get_data():
    # Loads the features dataset to a pandas dataframe and returns it
# Load the Diabetes dataset
    from feature_lb import df
    return df

#cross validation when we have 2 features
# evaluate a model with a given number of repeats
def evaluate_model(X, y, repeats):
	# prepare the cross-validation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=repeats, random_state=1)
	# create model
	model = linearRegression()
	# evaluate model
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # get the features
    #df = get_data()
    # 1- Split data to train and test
    df = get_data()
    train_X, test_X = train_test_split(df, test_size=0.2, random_state=42)
    #we don't have a second feature right now, so Y is empty
    train_Y, test_Y = [],[]
    print("train set of X:\n", train_X, "\n")
    print("test set of X:\n",test_X)
    # 2- Cross-fold validation on the train data
    #I can't do cross validation when we have only 1 feature
    
    # 3- Train a model
   #by changing this model name to get different model
    model_name = 'linear_regression'
    if model_name == 'linear_regression':
       res = linear_regression(train_X,train_Y)
    elif model_name == 'decision_tree':
        res = train_decision_tree()
    elif model_name == 'naive_bayes':
        res = train_naive_bayes()
    elif model_name == 'random_forest':
        res = train_random_forest()
      
    print("linear regression's result:", res)
    # 4- print the results
