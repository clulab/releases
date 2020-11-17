from __future__ import print_function
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import scipy.stats as stat
from sklearn.model_selection import GridSearchCV
from imblearn.metrics import classification_report_imbalanced
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
import numpy as np
import string
import random
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
import sklearn.metrics as skm
from skmultilearn.utils import measure_per_label
from sklearn.model_selection import StratifiedKFold
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

random.seed(a=42)
def stds_from_mean(ytd, std, mean, how_many_std_not_to_bin):
    # how_many_std_not_to_bin - if the dist from the mean is under this number, then the distance in stds is the distance component of the label; if distance is equal to or more than this number of stds, this number of stds is where all those infrequent datapoints are binned to; e.g., if the mean=2, std = 1.6, and the datapoint's time to discovery is 3, the datapoint is within 1 std away from the mean (that's why "+ 1" in the distance from the mean calculation) and it's above the mean, so the assigned sign is "+"; if the time to discovery is 30 years, that's between 17 and 18 stds from the mean and we don't want that many bins---those are outliers, so we we will use how_many_std_not_to_bin to decide what class to assign to outliers; with how_many_std_not_to_bin = 4, for example, any datapoint with how_many_std_not_to_bin >= 4 will be assigned to class 4 (and the corresponding +- sign)
    if ytd != 10000:
        direction = "+" if ytd > mean else "-"
        dist_from_mean = int((ytd - mean)/std) + 1

        if dist_from_mean < how_many_std_not_to_bin:
            label = direction + str(dist_from_mean)
            # print(label)
        else:
            label = direction + str(how_many_std_not_to_bin)
            # print("here ", label)
    else:
        label = "None"
    # print("ytd: ", ytd, " label: ", label)
    return label


def return_label_per_ytd(ytd, std, mean):
    label = stds_from_mean(ytd, std, mean, 3)
    # print(label)
    return label


def convert_years_to_discovery_to_class_labels(years_to_discovery, std, mean):
    # print("HERE: ", years_to_discovery, "<-")
    # std and mean could be calculated right here in the script?
    # classes are +- std from mean
    for ytd in years_to_discovery:
        # print(ytd, " ", return_label_per_ytd(ytd, std, mean))
        yield return_label_per_ytd(ytd, std, mean)


X_train_pos = np.loadtxt(open("/home/alexeeva/Repos/textgraphs2018-discovery-local/textgraphs2018-discovery/toy_data/split_batch_5_positive_datapoints_SMALLER_GRAPH_with_all_features-2020-10-17.csv", "rb"), delimiter="\t",usecols=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22, 23, 24, 25, 26, 27), skiprows=1)#
#X_train = np.loadtxt(open("train_dev.csv", "rb"), delimiter=",",usecols=(0,1,2,3,4))
y_train_pos = list(np.loadtxt(open("/home/alexeeva/Repos/textgraphs2018-discovery-local/textgraphs2018-discovery/toy_data/split_batch_5_positive_datapoints_SMALLER_GRAPH_with_all_features-2020-10-17.csv", "rb"), delimiter="\t",dtype=int,usecols=28, skiprows=1))

# print(y_train_pos)

X_train_neg = np.loadtxt(open("/home/alexeeva/Repos/textgraphs2018-discovery-local/textgraphs2018-discovery/toy_data/split_batch_5_negative_datapoints_SMALLER_GRAPH_with_all_features-2020-10-21.csv", "rb"), delimiter="\t",usecols=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22, 23, 24, 25, 26, 27), skiprows=1)#
#X_train = np.loadtxt(open("train_dev.csv", "rb"), delimiter=",",usecols=(0,1,2,3,4))
y_train_neg = list(np.loadtxt(open("/home/alexeeva/Repos/textgraphs2018-discovery-local/textgraphs2018-discovery/toy_data/split_batch_5_negative_datapoints_SMALLER_GRAPH_with_all_features-2020-10-21.csv", "rb"), delimiter="\t",dtype=int,usecols=28, skiprows=1))

# print(y_train_neg)


all_X_pre_shuffle = np.concatenate((X_train_pos, X_train_neg))
all_y_pre_shuffle = np.concatenate((y_train_pos, y_train_neg))
#
X, y = shuffle(all_X_pre_shuffle, all_y_pre_shuffle)
# print(X)
# print(y)
# print(type(y_train))
# print(type(y_train[0]))

print(y)

y_without_thousands = [y_item for y_item in y if y_item < 1000]
y_train_as_array = np.array(y_without_thousands)
mean = np.mean(y_train_as_array)
print("mean: ", mean)
std = np.std(y_train_as_array)
print("std: ", std)

y_train = list(convert_years_to_discovery_to_class_labels(y, std, mean))


labels = list(set(y_train))
print(labels)
# for label in labels:
#     print(label, type(label))
y_train_full = np.array(y_train)
print(y_train_full)

X_train, X_test, y_train, y_test = train_test_split(X, y_train_full, test_size=0.20, random_state=42)




# X_test = np.loadtxt(open("/media/alexeeva/ee9cacfc-30ac-4859-875f-728f0764925c/storage/IndepStudyGHPSpring2020/InfluenceGraph/paths/abc_1000_nodes_dict-based_paths-apr28_el_gato_with_all_features_test_with_negative_examples.csv", "rb"), delimiter=",",usecols=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22), skiprows=1)#
#
# #X_test = np.loadtxt(open("test.csv", "rb"), delimiter=",",usecols=(0,1,2,3,4))
# y_test = np.loadtxt(open("/media/alexeeva/ee9cacfc-30ac-4859-875f-728f0764925c/storage/IndepStudyGHPSpring2020/InfluenceGraph/paths/abc_1000_nodes_dict-based_paths-apr28_el_gato_with_all_features_test_with_negative_examples.csv", "rb"), delimiter=",",dtype=int,usecols=23, skiprows=1)
#
# y_test = list(convert_years_to_discovery_to_class_labels(y_test, std, mean))
# node text; the delimeter tells the loader which columns to take into account; i dont have this yet
# text_test= np.loadtxt(open("/media/alexeeva/ee9cacfc-30ac-4859-875f-728f0764925c/storage/IndepStudyGHPSpring2020/InfluenceGraph/paths/abc_1000_nodes_dict-based_paths-apr28_el_gato_with_all_features_test.csv", "rb"), delimiter='"',dtype=str,usecols=(1,3,5), skiprows=1)
# A_idf = np.loadtxt(open("/media/alexeeva/ee9cacfc-30ac-4859-875f-728f0764925c/storage/IndepStudyGHPSpring2020/InfluenceGraph/paths/abc_1000_nodes_dict-based_paths-apr28_el_gato_with_all_features_test_with_negative_examples.csv", "rb"), delimiter=",",usecols=13, skiprows=1)
# C_idf = np.loadtxt(open("/media/alexeeva/ee9cacfc-30ac-4859-875f-728f0764925c/storage/IndepStudyGHPSpring2020/InfluenceGraph/paths/abc_1000_nodes_dict-based_paths-apr28_el_gato_with_all_features_test_with_negative_examples.csv", "rb"), delimiter=",",usecols=14, skiprows=1)

# with open("/media/alexeeva/ee9cacfc-30ac-4859-875f-728f0764925c/storage/IndepStudyGHPSpring2020/InfluenceGraph/paths/abc_1000_nodes_dict-based_paths-apr28_el_gato_with_all_features_test_with_negative_examples.csv", 'r') as test:
    # content = [t.rstrip("\n") for t in test]
    # r3_id_test = [x.split(' ')[-1] for x in content]
    # r3_label_test = [x.split('"')[-2] for x in content]


X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)


names = [
        # "Neural Net",
        # "AdaBoost"  #,
        # "Random Forest" #,
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
              # MLPClassifier()
              # AdaBoostClassifier(DecisionTreeClassifier(max_depth=40,random_state=23,class_weight="balanced"))   #,
              # RandomForestClassifier(n_jobs=-1) #,
               LinearSVC(dual=False,fit_intercept=False, verbose=0) #todo: add the multiclass arg explicitly
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
              # [{'hidden_layer_sizes':[(300,),(800,),(1500,)],'early_stopping':[True], 'learning_rate': ['adaptive'], 'learning_rate_init':[1], 'max_iter': [10000000], 'momentum': [0.1,0.5,0.9], 'solver': ['sgd'],'random_state':[10008]}]
#               ,
               # [{'hidden_layer_sizes':[(300,)],'early_stopping':[True], 'learning_rate': ['adaptive'], 'learning_rate_init':[1], 'max_iter': [10000000], 'momentum': [0.1], 'solver': ['sgd'],'random_state':[10008]}]
#
               #AdaBoostClassifier
              # [{'random_state':[10008],'n_estimators': [600]}]
#               ,
#
#               #RandomForestClassifier
              # [{'max_depth': [5,None], 'max_features':[1,None,'auto'],'n_estimators': [10,50,100],'class_weight':[None,'balanced'],'random_state':[10008]}]
#               ,
#
              # SVC(kernel='linear')
               [{'C': [0.025, 1, 10, 100],'class_weight':[None,'balanced'],'random_state':[10008], 'multi_class':['ovr']}]
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
    grid = GridSearchCV(clf, param_grid=param_grid,scoring='f1_micro')  #scoring='f1', scoring=scorer
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


    all_predictions = []
    all_gold = []

    k = 5
    for i in range(k):
        X_train, X_test, y_train, y_test = train_test_split(X, y_train_full, test_size=0.20, stratify=y_train_full)
        # clf = AdaBoostClassifier(n_estimators=grid.best_params_['n_estimators'], random_state=grid.best_params_['random_state'])
        # clf = MLPClassifier(hidden_layer_sizes=grid.best_params_['hidden_layer_sizes'], early_stopping=grid.best_params_['early_stopping'], learning_rate=grid.best_params_['learning_rate'], learning_rate_init=grid.best_params_['learning_rate_init'], max_iter=grid.best_params_['max_iter'], momentum=grid.best_params_['momentum'], solver=grid.best_params_['solver'], random_state=grid.best_params_['random_state'])


        # clf = RandomForestClassifier(max_depth=grid.best_params_['max_depth'], max_features=grid.best_params_['max_features'],n_estimators=grid.best_params_['n_estimators'],class_weight=grid.best_params_['class_weight'],random_state=grid.best_params_['random_state'])

        clf = LinearSVC(dual=False,fit_intercept=False, verbose=0, C=grid.best_params_['C'],class_weight=grid.best_params_['class_weight'],random_state=grid.best_params_['random_state'],multi_class=grid.best_params_['multi_class'])
        clf.fit(X_train, y_train)
        y_true, y_pred = y_test, clf.predict(X_test)
        all_predictions = np.append(all_predictions, y_pred)
        all_gold = np.append(all_gold, y_true)
        print(classification_report_imbalanced(y_true, y_pred))


print("FINAL REPORT:")
print(all_gold)
print(all_predictions)
print(classification_report_imbalanced(all_gold, all_predictions))


#     all_predictions = []
#     all_gold = []
#
#     skf = StratifiedKFold(n_splits=5)
#     skf.get_n_splits(X, y)
#     print(skf)
#     for train_index, test_index in skf.split(X, y):
#         # print("TRAIN:", train_index, "TEST:", test_index)
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         clf = AdaBoostClassifier(n_estimators=grid.best_params_['n_estimators'], random_state=grid.best_params_['random_state'])
#         clf.fit(X_train, y_train)
#         y_true, y_pred = y_test, clf.predict(X_test)
#         print(type(y_true))
#         print(type(y_pred))
#         print(y_pred)
#         np.append(all_predictions, y_pred)
#         np.append(all_gold, y_true)
#         print(classification_report_imbalanced(y_true, y_pred))
#
# print("FINAL REPORT:")
# print(all_predictions)
# print(classification_report_imbalanced(all_gold, all_predictions))
        # per_lab = measure_per_label(skm.accuracy_score, y_true, y_pred)
        # print(per_lab)

    # scores = cross_validate(grid, X_train, y_train, scoring="f1_micro",
    #                     cv=5, return_train_score=True, verbose=1)
    # print("here: ", scores['test_score'])
    # # print("scores: ", scores)
    #
    # per_lab = measure_per_label(skm.accuracy_score, y_true, y_pred)
    # print(per_lab)
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
