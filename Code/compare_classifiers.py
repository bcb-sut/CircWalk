from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, roc_curve, auc, \
     roc_curve, roc_auc_score
from sklearn.utils import shuffle
# fŭr normalization
from sklearn.preprocessing import normalize, scale
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pathlib
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import xgboost as xgb
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

pathlib.Path("../Results/" + "Comparison").mkdir(parents=True, exist_ok=True)


benchmark = pd.read_csv("../Data/clean_benchmark.csv", header=None)
benchmark = benchmark.values.tolist()
print("Method: " + "Comparison")
x = []
y = []
for i, b in enumerate(benchmark):
    circ = b[0]
    dis = b[1]
    label = b[2]
    x.append([circ, dis])
    y.append(label)

x, y = shuffle(x, y, random_state=7997)
x = np.array(x)
y = np.array(y)
n = 5
kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=7997)


classifier_dict = {
    "Adaboost with RF": AdaBoostClassifier(base_estimator=RandomForestClassifier(max_depth=10), learning_rate=10 ** -2),
    "RF": RandomForestClassifier(random_state=28, n_estimators=100),
    "Multilayer Perceptron": MLPClassifier(solver='adam', alpha=1e-2, hidden_layer_sizes=(10, 2), max_iter=1000,
                                           random_state=1),
    "Logistic Regression": LogisticRegression(C=10, max_iter=1000),
    "SVM": svm.SVC(kernel="linear", C=5, probability=True)
    }

optimal_feature_size = {
    "Adaboost with RF": 10,
    "RF": 10,
    "Multilayer Perceptron": 10,
    "Logistic Regression": 80,
    "SVM": 90,
    "XGBoost": 20
}


pretty_methods = {"Adaboost with RF": "ABRF", "XGBoost": "XGB", "Multilayer Perceptron": "MP",
                  "Logistic Regression": "LR", "RF": "RF", "SVM": "SVM"}

fold_num = 0
for train_ix, test_ix in kfold.split(x, y):
    train, test = x[train_ix], x[test_ix]
    y_train, y_test = y[train_ix], y[test_ix]
    for method in optimal_feature_size.keys():
        print(method)
        embed_size = optimal_feature_size[method]
        df_embeddings = pd.read_csv("../Data/entity_representations" + str(embed_size) + ".embeddings", header=None,
                                    sep=" ")
        df_embeddings.columns = ["id"] + ["val" + str(i) for i in range(1, df_embeddings.shape[1])]
        df_embeddings['alle'] = [tuple(x) for x in
                                 df_embeddings[
                                     ["val" + str(i) for i in range(1, df_embeddings.shape[1])]].values.tolist()]
        # print(df_embeddings)
        embeddings = dict(zip(df_embeddings.id, df_embeddings.alle))
        temp = train.copy()
        method_specific_train = []
        for i, b in enumerate(temp):
            circ = embeddings[b[0]]
            dis = embeddings[b[1]]
            method_specific_train.append([circ[circiter] for circiter in range(len(circ))] + [dis[disiter] for disiter in range(len(dis))])

        temp = test.copy()
        method_specific_test = []
        for i, b in enumerate(temp):
            circ = embeddings[b[0]]
            dis = embeddings[b[1]]
            method_specific_test.append([circ[circiter] for circiter in range(len(circ))] + [dis[disiter] for disiter in range(len(dis))])
        method_specific_train = np.array(method_specific_train)
        method_specific_test = np.array(method_specific_test)
        # train = np.array([list(embeddings[train[t][0]] + embeddings[train[t][1]]) for t in range(len(train))])
        # test = np.array([list(embeddings[test[t][0]] + embeddings[test[t][1]]) for t in range(len(test))])

        if not method == "XGBoost":
            clf = classifier_dict[method]
            clf.fit(method_specific_train, y_train)
            fpr, tpr, thresh = roc_curve(y_test, clf.predict_proba(method_specific_test)[:, 1])
            aucc = roc_auc_score(y_test, clf.predict_proba(method_specific_test)[:, 1])
            aucc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (str(pretty_methods[method]), float(aucc * 100)))

        else:
            dtrain = xgb.DMatrix(method_specific_train, label=y_train)
            param = {
                'eta': 0.3,
                'max_depth': 7,
                'objective': 'multi:softprob',
                'num_class': 2}
            steps = 50  # The number of training iterations
            dtest = xgb.DMatrix(method_specific_test, label=y_test)
            evallist = [(dtrain, 'train')]
            num_round = 64
            bst = xgb.train(param, dtrain, num_round, evallist)
            y_pred = bst.predict(dtest, ntree_limit=bst.best_iteration + 1)
            fpr, tpr, _ = roc_curve(y_test, y_pred[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (str(pretty_methods[method]), float(roc_auc * 100))) #lw = 2
            name = "circwalk"
            pathlib.Path("../Data/" + str(embed_size)).mkdir(parents=True, exist_ok=True)
            df_roc = pd.DataFrame(np.column_stack([fpr, tpr]), columns=['fpr', 'tpr'])
            df_roc.to_excel(
                "../Data/" + "points" + "/" + name + "-fold-" + str(
                    fold_num + 1) + ".xls", index=False, header=True)

            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="best")
            plt.title("ROC for Fold " + str(fold_num + 1))
            pathlib.Path("../Results/" + "Comparison").mkdir(parents=True, exist_ok=True)
            plt.savefig("../Results/" + "Comparison" + "/Fold" + str(fold_num + 1) + ".png")
            plt.close()
            print(str(fold_num + 1))
            fold_num += 1