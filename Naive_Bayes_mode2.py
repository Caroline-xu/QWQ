import pandas as pd
import numpy as np

class Gaussian_NB():
    def __init__(self):
        self.num_of_samples = None
        self.num_of_class = None
        self.class_name = []
        self.prior_prob = []
        self.X_mean = []
        self.X_var = []

    def SepByClass(self, X, y):
        ###Separate data by category###
        ###Input unclassified features and targets, output classified data (dictionary form)###
        self.num_of_samples=len(y) #total sample number
        
        y=y.reshape(X.shape[0],1)
        data=np.hstack((X,y)) #Combine features and targets into complete data
        
        data_byclass={} #Initialize classification data as an empty dictionary
        #Extract all kinds of data. The key of the dictionary is the category name and the value is the corresponding classification data
        for i in range(len(data[:,-1])):
            if i in data[:,-1]:
                data_byclass[i]=data[data[:,-1]==i]
        
        self.class_name=list(data_byclass.keys()) #Category name
        self.num_of_class=len(data_byclass.keys()) #The total number of categories
        
        return data_byclass
    
    def CalPriorProb(self, y_byclass):
        ###Calculate the prior probability of y (using Laplacian smoothing)###
        ###Input the target under the current category, output the prior probability of the target###
        #Calculation formula :(number of samples under current category +1)/(total number of samples + total number of categories)
        return (len(y_byclass)+1)/(self.num_of_samples+self.num_of_class)
    
    def CalXMean(self, X_byclass):
        ### Calculate the average value of each dimension ###
        ### Input the feature under the current category, output the average value of each dimension of this feature ###
        X_mean=[]
        for i in range(X_byclass.shape[1]):
            X_mean.append(np.mean(X_byclass[:,i]))
        return X_mean

    def CalXVar(self, X_byclass):
        ### Calculate the variance of each dimension of each category feature 
        ###Input the feature under the current category and output the variance of each dimension of the feature ###
        X_var=[]
        for i in range(X_byclass.shape[1]):
            X_var.append(np.var(X_byclass[:,i]))
        return X_var
    
    def CalGaussianProb(self, X_new, mean, var):
        ### Calculate the conditional probabilities of the training set features (conforming to normal distribution) in each class 
        ###Input the features of the new sample, the mean value and variance of the features of the training set, and output the 
        # distribution probability of the features of the new sample in the corresponding training set ###
        #计算公式：(np.exp(-(X_new-mean)**2/(2*var)))*(1/np.sqrt(2*np.pi*var))
        gaussian_prob=[]
        for a,b,c in zip(X_new, mean, var):
            formula1=np.exp(-(a-b)**2/(2*c))
            formula2=1/np.sqrt(2*np.pi*c)
            gaussian_prob.append(formula2*formula1)
        return gaussian_prob

    def fit(self, X, y):
        ### Training data ###
        ### Input training set features and targets, the prior probability of output targets, the mean and variance of features 
        ###convert input X,y to numpy array
        X, y = np.asarray(X, np.float32), np.asarray(y, np.float32)      
        
        data_byclass=Gaussian_NB.SepByClass(X,y) #Categorize data
        #The target prior probability, characteristic mean and variance of all kinds of data are calculated
        for data in data_byclass.values():
            X_byclass=data[:,:-1]
            y_byclass=data[:,-1]
            self.prior_prob.append(Gaussian_NB.CalPriorProb(y_byclass))
            self.X_mean.append(Gaussian_NB.CalXMean(X_byclass))
            self.X_var.append(Gaussian_NB.CalXVar(X_byclass))
        
        return self.prior_prob, self.X_mean, self.X_var
        
    def predict(self,X_new):
        ###Forecast data###
        ###Input the characteristics of the new sample, output the most likely target of the new sample###
        #Converts the input X_new to a NUMpy array
        X_new=np.asarray(X_new, np.float32)
        
        posteriori_prob=[] #Initialize the maximum posteriori probability
        
        for i,j,o in zip(self.prior_prob, self.X_mean, self.X_var):
            gaussian=Gaussian_NB.CalGaussianProb(X_new,j,o)
            posteriori_prob.append(np.log(i)+sum(np.log(gaussian)))
            idx=np.argmax(posteriori_prob)
        
        return self.class_name[idx]

def train_naive_bayes(X_train, X_test, y_train, y_test):  
     #Gaussian_NB
    Gaussian_NB_model = Gaussian_NB()
    Gaussian_NB_model.fit(X_train,y_train) #Use Gaussian_NB model to train data
    acc=0
    TP=0
    FP=0
    FN=0
    for i in range(len(X_test)):
        predict=Gaussian_NB_model.predict(X_test.iloc[i,:])
        target=np.array(y_test)[i]
        if predict==1 and target==1:
            TP+=1
        if predict==0 and target==1:
            FP+=1
        if predict==target:
            acc+=1
        if predict==1 and target==0:
            FN+=1
    print("accuracy rate:",acc/len(X_test))
    print("precision:",TP/(TP+FP))
    print("recall:",TP/(TP+FN))
    print("F1:",2*TP/(2*TP+FP+FN))