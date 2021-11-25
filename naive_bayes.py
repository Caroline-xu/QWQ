def train_naive_bayes(X_train, X_test, y_train, y_test):
    
    from sklearn import datasets
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import GridSearchCV
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
    from sklearn.metrics import f1_score
    from dummy_classifier import dummy_classifier
    #because naive bayes has no parameter to tune, so it could not use grid search
    regressor = GaussianNB()
    regressor.fit(X_train,y_train)
    y_pred =regressor.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    #print("Detailed classification report:")
    #print(classification_report(y_test, y_pred))
    acc = accuracy_score(y_test,y_pred)# need to change after this
    mse = mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mae)
    #r2Score=r2_score(y_test, y_pred)
    macroF1=f1_score(y_test, y_pred, average='macro')
    microF1=f1_score(y_test, y_pred, average='micro')
    weightedF1=f1_score(y_test, y_pred, average='weighted')
    dummy=dummy_classifier(X_train, y_train)
    return acc, mse, mae, rmse, macroF1,microF1,weightedF1,dummy

    #acc = np.sum(y_test == y_pred) / X_test.shape[0]
    #print("Test Acc : %.3f" % acc)
    # predict the model
    #y_proba = gnb.predict_proba(X_test[:1])
    #print(gnb.predict(X_test[:1]))
    #print("The estimated probability:", y_proba)
