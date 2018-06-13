from __future__ import print_function

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import scipy.stats as stat
from sklearn.model_selection import GridSearchCV
from imblearn.metrics import classification_report_imbalanced
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from mimport metrics
import numpy as np
import string
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
#from sklearn.naive_bayes import GaussianNB
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from imblearn.metrics import (geometric_mean_score, make_index_balanced_accuracy)
#from imblearn.under_sampling import NearMiss
#from imblearn.metrics import geometric_mean_score as gmean
#from imblearn.metrics import make_index_balanced_accuracy as iba

X_train = np.loadtxt(open("train_dev.csv", "rb"), delimiter=",",usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22))
#X_train = np.loadtxt(open("train_dev.csv", "rb"), delimiter=",",usecols=(0,1,2,3,4))
y_train = np.loadtxt(open("train_dev.csv", "rb"), delimiter=",",dtype=int,usecols=23)

X_test = np.loadtxt(open("test.csv", "rb"), delimiter=",",usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22))
#X_test = np.loadtxt(open("test.csv", "rb"), delimiter=",",usecols=(0,1,2,3,4))
y_test = np.loadtxt(open("test.csv", "rb"), delimiter=",",dtype=int,usecols=23)

text_test= np.loadtxt(open("test.csv", "rb"), delimiter='"',dtype=str,usecols=(1,3,5))
A_idf = np.loadtxt(open("test.csv", "rb"), delimiter=",",usecols=0)
C_idf = np.loadtxt(open("test.csv", "rb"), delimiter=",",usecols=2)

with open("test.csv", 'r') as test:
    content = [t.rstrip("\n") for t in test]
    r3_id_test = [x.split(' ')[-1] for x in content]
    r3_label_test = [x.split('"')[-2] for x in content]


    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)


    names = [
#             "Neural Net",
#             "AdaBoost"  #,
#             "Random Forest" #,
             "SVM with linear kernel" #,
#             "SVM with polynomial kernel"
             
             # ,
             # "Naive Bayes",
             # "QDA",
             # "Decision Tree",
             # "Logistic Regression"
             # "Nearest Neighbors",
             ]

    classifiers = [
#                   MLPClassifier(),
#                   AdaBoostClassifier(DecisionTreeClassifier(max_depth=40,random_state=23,class_weight="balanced"))   #,
#                   RandomForestClassifier() #,
                   SVC(kernel='linear')#,
#                   SVC(kernel='poly')
                   
                   #,
                   # GaussianNB(),
                   # QuadraticDiscriminantAnalysis(),
                   # DecisionTreeClassifier(),
                   # LogisticRegression(solver='liblinear',dual=False,fit_intercept=False),
                   # KNeighborsClassifier(n_jobs=-1)
                   ]



    param_grids = [
    #               #MLPClassifier
#                   [{'hidden_layer_sizes':[(300,),(800,),(1500,)],'early_stopping':[True], 'learning_rate': ['adaptive'], 'learning_rate_init':[1], 'max_iter': [10000000], 'momentum': [0.1,0.5,0.9], 'solver': ['sgd'],'random_state':[10008]}]
    #               ,
                   
                   #AdaBoostClassifier
#                   [{'random_state':[10008],'n_estimators': [600]}]
    #               ,
    #
    #               #RandomForestClassifier
#                   [{'max_depth': [5,None], 'max_features':[1,None,'auto'],'n_estimators': [10,50,100],'class_weight':[None,'balanced'],'random_state':[10008]}]
    #               ,
    #
    #               #SVC(kernel='linear')
                   [{'C': [0.025, 1, 10, 100],'class_weight':[None,'balanced'],'random_state':[10008]}]
    #               ,
    #
    #               #SVC(kernel='poly')
#                    [{'C': [0.025, 1, 10, 100],'class_weight':[None,'balanced'],'random_state':[10008]}]
    #               #,
                   
                   # [{'n_neighbors': [5,10,20],'weights':['uniform','distance']}],
                   # [{'max_depth': [5,None], 'max_features':[1,None,'auto'],'class_weight':[None,'balanced']}],
                   # [{'alpha': [0.00001], 'activation': ['logistic'], 'solver': ['adam'], 'random_state': [830], 'learning_rate': ['constant'], 'max_iter': [100]}],
                   # [{}],
                   # [{}],
                   #
                   ]

    # gmean = iba(alpha=0.1, squared=True)(gmean)
    # scorer = metrics.make_scorer(gmean)

    # iterate over classifiers
    for name, clf, param_grid in zip(names, classifiers,param_grids):
        print("# classifier: %s" % name)
        print()
        grid = GridSearchCV(clf, param_grid=param_grid,scoring='f1')  #scoring='f1', scoring=scorer
        grid.fit(X_train, y_train)
        print()
        print("Best parameters set found on development set:")
        print()
        print(grid.best_params_)

        print()
        print("Scores for all configurations:")
        for i in range(len(grid.cv_results_['mean_test_score'])):
            print("params:", grid.cv_results_['params'][i])
            print("mean_test_score:", grid.cv_results_['mean_test_score'][i])
            print("std_test_score:", grid.cv_results_['std_test_score'][i])
            print("----")

            print()
            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
        y_true, y_pred = y_test, grid.predict(X_test)
        print(classification_report_imbalanced(y_true, y_pred))

#    y_proba = grid.predict_proba(X_test)
#    with open('classifiers_probabilities.tsv', 'w') as output:
#     output.write("pro_0\tpro_1\tpredict\tgold\tconceptA\tconceptB\tconceptC\tA-idf\tC-idf\tnew finding\tgold-r3-id\tgold-r3-polarity\n")
#     for i in range(len(y_test)):
#         output.write(str(y_proba[i][0]))
#         output.write('\t')
#         output.write(str(y_proba[i][1]))
#         output.write('\t')
#         output.write(str(y_pred[i]))
#         output.write('\t')
#         output.write(str(y_true[i]))
#         output.write('\t')
#         output.write(str(text_test[i][0]))
#         output.write('\t')
#         output.write(str(text_test[i][1]))
#         output.write('\t')
#         output.write(str(text_test[i][2]))
#         output.write('\t')
#         output.write(str(A_idf[i]))
#         output.write('\t')
#         output.write(str(C_idf[i]))
#         output.write('\t')
#         output.write(str(text_test[i][0])+" -> "+str(text_test[i][2]))
#         if(y_true[i]):
#             output.write('\t')
#             output.write(str(r3_id_test[i]))
#             output.write('\t')
#             output.write(str(r3_label_test[i]))
#         output.write('\n')
#    print()


