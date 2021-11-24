from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm

def find_best_param(X_train, X_test, y_train, y_test):
    scores = ["precision", "recall"]
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(X_train,y_train)
    GridSearchCV(estimator=svc,param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    y_true, y_pred = y_test, clf.predict(X_test)
    print("Detailed classification report:")
    print(classification_report(y_true, y_pred))
    
       