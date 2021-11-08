import pandas as pd
from linearRegression import linear_regression
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
# train_linear_regression() is defined in linear_regression.py
#from linear_regression import train_linear_regression
#from decision_tree import train_decision_tree
from Naive_Bayes import train_naive_bayes
#from random_forest import train_random_forest
def get_data():
<<<<<<< Updated upstream
    # Loads the features dataset to a pandas dataframe and returns it
# Load the Diabetes dataset
    from feature_lb import df
    return df
=======
    df = feature_lb()
    X = df.loc[:, df.columns != 'New Label']
    y = df['New Label']
    final_df = PCA_reduce_dimension(X,y)
    #print(df)
    return final_df
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
    #I can't do cross validation when we have only 1 feature
=======
    k = 5
    kf = KFold(n_splits=k, random_state=None)
    acc_score = []
    nb_acc_score = []

    for train_index , test_index in kf.split(X):
        X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]
        
        ### comment out the following line to train random forest classfication, and print out the accuracy 
        # train_random_forest(X_train, X_test, y_train, y_test)
        
        acc = train_linear_regression(X_train, X_test, y_train, y_test)
        acc_score.append(acc)

        #naive bayes model
        nb_acc = train_naive_bayes(X_train, X_test, y_train, y_test)
        nb_acc_score.append(nb_acc)

    avg_acc_score = sum(acc_score)/k
    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))

    #naive bayes model
    avg_nb_acc_score = sum(nb_acc_score)/k
    print('accuracy of each fold in naive bayes model - {}'.format(nb_acc_score))
    print('Avg accuracy of Naive Bayes Model: {}'.format(avg_nb_acc_score))
>>>>>>> Stashed changes
    
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
