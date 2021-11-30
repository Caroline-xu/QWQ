import pandas as pd
from feature_lb import feature_lb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# for standardize data
from sklearn.preprocessing import StandardScaler
# import PCA
from sklearn.decomposition import PCA
from linear_regression import train_linear_regression
from decision_tree import train_decision_tree
from naive_bayes import train_naive_bayes
from random_forest import train_random_forest

# call this function to normalize the data (scalling) and use PCA to reduce dimension(features) to 300 
# take X as features and Y as label (unnormailzied)

def PCA_reduce_dimension(X, Y):
    # this function is used to reduce dimension of the data frame to 300 columns
    # input parameter: X(data frame except label column), Y(label column)
    # return value: final_df (whole dataframe after PCA reduced dimension to 300 columns)
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=300)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents)
    finalDf = pd.concat([principalDf, Y], axis = 1)
    return finalDf

def get_data():
    df = feature_lb()
    X = df.loc[:, df.columns != 'New Label']
    y = df['New Label']
    final_df = PCA_reduce_dimension(X,y)
    #print(df)
    return final_df

def splitSet_forGrid(model_name):
    # this function will split set and do grid search(do cross validation) for
    # linear regression and naive bayes
    # parameter: model_name (e.g. "random_forest"")
    # output value: model name, accuracy score, macro F1 score, micro F1 score, weighted F1 score
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=25)
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=25)
    if model_name == "random_forest":
        acc, macroF1,microF1,weightedF1 = train_random_forest(X_train, X_test, y_train, y_test)
    elif model_name == "decision_tree":
        acc, macroF1,microF1,weightedF1 = train_decision_tree(X_train, X_test, y_train, y_test) 
        
    print("This model is :", model_name)
    print('Avg accuracy :',"%.2f" %acc)
    print('Avg macro F1 score :',"%.2f" % macroF1)
    print('Avg micro F1 score :',"%.2f" % microF1) 
    print('Avg weighted F1 score :',"%.2f" % weightedF1) 

def fiveCrossValidation(model_name):
    # this function will split set and do 5 folds cross validation for
    # linear regression and naive bayes
    # parameter: model_name (e.g. "linear_regression")
    # return value: model name, accuracy score, macro F1 score, micro F1 score, weighted F1 score
    k = 5 # 5 fold
    kf = KFold(n_splits=k, random_state=None)
    acc_total = []
    macroF1_total = []
    microF1_total = [] 
    weightedF1_total = []
    #dummy_total = []
    for train_index , test_index in kf.split(X):
        X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]
        
        if model_name == 'linear_regression':
            acc,macroF1,microF1,weightedF1 = train_linear_regression(X_train, X_test, y_train, y_test)
        elif model_name == 'naive_bayes':
            acc,macroF1,microF1,weightedF1 = train_naive_bayes(X_train, X_test, y_train, y_test)
            
        acc_total.append(acc)
        macroF1_total.append(macroF1)
        microF1_total.append(microF1)
        weightedF1_total.append(weightedF1)
        #dummy_total.append(dummy)
        
    avg_acc = sum(acc_total)/k
    avg_macroF1 = sum(macroF1_total)/k
    avg_microF1 = sum(microF1_total)/k 
    avg_weightedF1 = sum(weightedF1_total)/k
    #avg_dummy = sum(dummy_total)/k
    
    #print('accuracy of each fold - {}'.format(acc_total))
    print("This model is :", model_name)
    print('Avg accuracy :',"%.2f" %avg_acc)

    #print('Avg r2 score :',"%.2f" % avg_r2Score)
    print('Avg macro F1 score :',"%.2f" % avg_macroF1)
    print('Avg micro F1 score :',"%.2f" % avg_microF1) 
    print('Avg weighted F1 score :',"%.2f" % avg_weightedF1) 
    #print('dummy total:', dummy_total)
    #print('Avg dummy:',"%.2f" % avg_dummy)  
    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # get the features
    #df = get_data()
    df = feature_lb()
    # 1- Split data to train and test
    #df = get_data()
    #X is whole dataframe except label column 
    X = df.loc[:, df.columns != 'New Label']
    #y is label column
    y = df['New Label']
    
    ####train model by using 5 fold cross validation####
    # 2- Cross-fold validation on the train data   
    
    #model_name = "random_forest"
    #model_name = "linear_regression"
    #model_name = "naive_bayes"
    model_name = "decision_tree"
    
    if model_name == "random_forest" or "decision tree":
        splitSet_forGrid(model_name) 
    elif model_name == "linear_regression" or "naive_bayes":
        fiveCrossValidation(model_name)  


    
    
    
    
   
