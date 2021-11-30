
from sklearn.metrics import f1_score
from dummy_classifier import dummy_classifier
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import accuracy_score
def train_decision_tree(X_train, X_test, y_train, y_test):
    # train algorithm by using cvSearch's best parameter
    parameters = {'max_leaf_nodes': list(range(2, 100)), 
                  'min_samples_split': [2, 3, 4],
                  "criterion": ["gini", "entropy"]}
    
    regressor = DecisionTreeClassifier()
    search = GridSearchCV(regressor, parameters)
    search.fit(X_train,y_train)
    #GridSearchCV(estimator=svc,param_grid=parameters)
    print("Best parameters set found on development set:")
    print(search.best_params_)
    print("Best estimator set found on development set:")
    print(search.best_estimator_) 
    y_pred =search.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    #print("Detailed classification report:")
    #print(classification_report(y_test, y_pred))
    acc = accuracy_score(y_test,y_pred)# need to change after this

    macroF1=f1_score(y_test, y_pred, average='macro')
    microF1=f1_score(y_test, y_pred, average='micro')
    weightedF1=f1_score(y_test, y_pred, average='weighted')
    #dummy=dummy_classifier(X_train, y_train)
    return acc, macroF1,microF1,weightedF1
 
