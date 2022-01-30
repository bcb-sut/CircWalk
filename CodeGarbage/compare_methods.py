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

method = "Comparison"
pathlib.Path("../Results/" + method).mkdir(parents=True, exist_ok=True)
sys.stdout = open("../Results/" + method + "/" + str(method) + "Report.docx", "w")


def sen_and_spec(y_pred, y_real):
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return sensitivity, specificity


benchmark = pd.read_csv("../Data/clean_benchmark.csv", header=None)
benchmark = benchmark.values.tolist()
print("Method: " + method)
for embed_size in range(10, 55, 10):
    print("Embedding Size:", embed_size)
    df_embeddings = pd.read_csv("../Data/entity_representations" + str(embed_size) + ".embeddings", header=None, sep=" ")
    df_embeddings.columns = ["id"] + ["val" + str(i) for i in range(1, df_embeddings.shape[1])]
    df_embeddings['alle'] = [tuple(x) for x in
                             df_embeddings[["val" + str(i) for i in range(1, df_embeddings.shape[1])]].values.tolist()]
    # print(df_embeddings)
    embeddings = dict(zip(df_embeddings.id, df_embeddings.alle))
    # print(benchmark)
    x = []
    y = []
    for i, b in enumerate(benchmark):
        circ = embeddings[b[0]]
        dis = embeddings[b[1]]
        label = b[2]
        x.append([circ[circiter] for circiter in range(len(circ))] + [dis[disiter] for disiter in range(len(dis))])
        y.append(label)

    x, y = shuffle(x, y, random_state=7997)
    x = np.array(x)
    y = np.array(y)

    # weka_file = pd.DataFrame(x, columns=["Number" + str(i + 1) for i in range(len(x[0]))])
    # weka_file["class"] = y
    # weka_file.to_csv("../Weka/" + str(embed_size) + ".csv", index=None)
    # continue

    n = 5
    # x = normalize(x)  # fŭr normalization
    kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=7997)

    stats = {"Accuracy": [], "Precision": [], "Recall": [], "F1 score": [], "Sensitivity": [], "Specificity": [],
             "AUC": []}

    i = 0
    for train_ix, test_ix in kfold.split(x, y):
        print("----------------------------------------------------------------------------------------------")
        # train = x[:int(len(x) * i / n)] + x[int(len(x) * (i + 1) / n):]
        # y_train = y[:int(len(x) * i / n)] + y[int(len(x) * (i + 1) / n):]
        # test = x[int(len(x) * i / n):int(len(x) * (i + 1) / n)]
        # y_test = y[int(len(x) * i / n):int(len(x) * (i + 1) / n)]
        train, test = x[train_ix], x[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # clf = svm.SVC(kernel="linear", C=5)
        classifier_dict = {"Adaboost with RF": AdaBoostClassifier(base_estimator=RandomForestClassifier(max_depth=10), learning_rate=10**-2),
                           "RF": RandomForestClassifier(random_state=28,n_estimators=100),
                           "Multilayer Perceptron": MLPClassifier(solver='lbfgs', alpha=1e-2, hidden_layer_sizes=(10, 2), max_iter=1000, random_state=1),
                           "Logistic Regression": LogisticRegression(C=10, max_iter=1000),
                           "SVM": svm.SVC(kernel="linear", C=5, probability=True)}
        for name, instance in classifier_dict.items():
            clf = instance
            clf.fit(train, y_train)
            y_pred = clf.predict(test)
            fpr, tpr, thresh = roc_curve(y_test, clf.predict_proba(test)[:, 1])
            aucc = roc_auc_score(y_test, clf.predict_proba(test)[:, 1])
            aucc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (str(name), float(aucc * 100)))

        dtrain = xgb.DMatrix(train, label=y_train)
        param = {
            'eta': 0.3,
            'max_depth': 3,
            'objective': 'multi:softprob',
            'num_class': 2}
        steps = 20  # The number of training iterations
        dtest = xgb.DMatrix(test, label=y_test)
        evallist = [(dtrain, 'train')]
        num_round = 10
        bst = xgb.train(param, dtrain, num_round, evallist)
        y_pred = bst.predict(dtest, ntree_limit=bst.best_iteration + 1)
        fpr, tpr, _ = roc_curve(y_test, y_pred[:, 1])
        roc_auc = auc(fpr, tpr)
        lw = 2
        plt.plot(fpr, tpr, lw=lw, label='%s (AUC = %0.2f)' % (str("XGBoost"), float(roc_auc * 100)))

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="best")
        plt.title("ROC for Fold " + str(i + 1))
        pathlib.Path("../Results/" + method + "/" + str(embed_size)).mkdir(parents=True, exist_ok=True)
        plt.savefig("../Results/" + method + "/" + str(embed_size) + "/" + "/Fold" + str(i + 1) + ".png")
        plt.close()
        print(str(i + 1))
        i += 1

        # clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(max_depth=10), learning_rate=10**-2)
        # clf.fit(train, y_train)
        # y_pred = clf.predict(test)
        # sen, spec = sen_and_spec(y_pred, y_test)

        # p = plot_roc_curve(clf, test, y_test)
        # pathlib.Path("../Results/" + method + "/" + str(embed_size)).mkdir(parents=True, exist_ok=True)
        # plt.savefig("../Results/" + method + "/" + str(embed_size) + "/RocFold" + str(i + 1) + ".png")
        # plt.close()

        # fpr, tpr, thresh = roc_curve(y_test, clf.predict_proba(test)[:, 1])
        # auc = roc_auc_score(y_test, y_pred)
        # plt.plot(fpr, tpr, label='Fold %s ROC (area = %0.2f)' % (str(i + 1), auc))

        # fold_stats = {"Fold Num": (i + 1)/100,
        #               "Accuracy": accuracy_score(y_test, y_pred),
        #               "Precision": precision_score(y_test, y_pred),
        #               "Recall": recall_score(y_test, y_pred),
        #               "F1 score": f1_score(y_test, y_pred),
        #               "Sensitivity": sen,
        #               "Specificity": spec,
        #               "AUC": auc}
        # fold_stats = {k: round(v*100, 3) for k, v in fold_stats.items()}
        # for k, v in fold_stats.items():
        #     print(k + ":", v)
        # for k in stats:
        #     stats[k].append(fold_stats[k])
        # print("----------------------------------------------------------------------------------------------")
        # print()
        # i += 1
    # plt.legend(loc="best")
    # pathlib.Path("../Results/" + method + "/" + str(embed_size)).mkdir(parents=True, exist_ok=True)
    # plt.savefig("../Results/" + method + "/" + str(embed_size) + "/RocAllFolds" + ".png")
    # plt.close()
    # print("Overall:")
    # for k, v in stats.items():
    #     print(k + ":", np.mean(v))
    # print()
    # print("###############################################################################")
    # print()

sys.stdout.close()
