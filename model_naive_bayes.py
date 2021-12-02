def train_naive_bayes(X_train, X_test, y_train, y_test):

    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from dummy_classifier import dummy_classifier
    #because naive bayes has no parameter to tune, so it could not use grid search
    regressor = GaussianNB()
    regressor.fit(X_train,y_train)
    y_pred =regressor.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    #print("Detailed classification report:")
    #print(classification_report(y_test, y_pred))
    acc = accuracy_score(y_test,y_pred)

    #r2Score=r2_score(y_test, y_pred)
    macroF1=f1_score(y_test, y_pred, average='macro')
    microF1=f1_score(y_test, y_pred, average='micro')
    weightedF1=f1_score(y_test, y_pred, average='weighted')
    #dummy=dummy_classifier(X_train, y_train)
    return acc, macroF1,microF1,weightedF1
