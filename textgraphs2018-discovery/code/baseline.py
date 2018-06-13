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
from sklearn import metrics
import numpy as np
import string, random

X_train = np.loadtxt(open("train_dev.csv", "rb"), delimiter=",",usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22))
y_train = np.loadtxt(open("train_dev.csv", "rb"), delimiter=",",dtype=int,usecols=23)

X_test = np.loadtxt(open("test.csv", "rb"), delimiter=",",usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22))
y_test = np.loadtxt(open("test.csv", "rb"), delimiter=",",dtype=int,usecols=23)

X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)

num_train = y_train.shape[0]
num_train_pos = y_train.sum()
ratio = num_train_pos * 1.0 / num_train

num_test = y_test.shape[0]
num_test_pos = int(num_test * ratio)
y_pred = np.array([0] * (num_test - num_test_pos) + [1] * num_test_pos)
random.seed(10008)
random.shuffle(y_pred)

print(classification_report_imbalanced(y_test, y_pred))

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


