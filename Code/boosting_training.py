from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.utils import shuffle
# fŭr normalization
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pathlib
import xgboost as xgb
import sys

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

method = "xgboost"
pathlib.Path("../Results/" + method).mkdir(parents=True, exist_ok=True)
sys.stdout = open("../Results/" + method + "/" + str(method) + "Report.docx", "w")


def sen_and_spec(y_pred, y_real):
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return sensitivity, specificity


output = []

benchmark = pd.read_csv("../Data/clean_benchmark.csv", header=None)
benchmark = benchmark.values.tolist()
print("Method: " + method)
x_auc = []
y_auc = []
for embed_size in range(10, 205, 10):
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
    n = 5
    # x = normalize(x)  # fŭr normalization
    kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=7997)

    stats = {"Accuracy": [], "Precision": [], "F1 score": [], "Sensitivity": [], "Specificity": [], "AUC": []}

    i = 0
    for train_ix, test_ix in kfold.split(x, y):
        print("----------------------------------------------------------------------------------------------")
        # train = x[:int(len(x) * i / n)] + x[int(len(x) * (i + 1) / n):]
        # y_train = y[:int(len(x) * i / n)] + y[int(len(x) * (i + 1) / n):]
        # test = x[int(len(x) * i / n):int(len(x) * (i + 1) / n)]
        # y_test = y[int(len(x) * i / n):int(len(x) * (i + 1) / n)]
        train, test = x[train_ix], x[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        dtrain = xgb.DMatrix(train, label=y_train)
        param = {
            'eta': 0.3,
            'max_depth': 7,
            'objective': 'multi:softprob',
            'num_class': 2}
        steps = 20  # The number of training iterations
        dtest = xgb.DMatrix(test, label=y_test)
        evallist = [(dtrain, 'train')]
        num_round = 64
        bst = xgb.train(param, dtrain, num_round, evallist)
        y_pred = bst.predict(dtest, ntree_limit=bst.best_iteration+1)

        fpr, tpr, _ = roc_curve(y_test, y_pred[:, 1])
        roc_auc = auc(fpr, tpr)
        # xgb.plot_importance(gbm)
        # plt.show()
        plt.figure()
        lw = 2

        roc_df = pd.DataFrame(
    {'fpr': fpr,
     'tpr': tpr
    })
        roc_df.to_excel("../Data/points/circwalk-fold-" + str(i + 1) + ".xls", index=False)

        plt.plot(fpr, tpr, lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="best")
        # p = plot_roc_curve(clf, test, y_test)
        pathlib.Path("../Results/" + method + "/" + str(embed_size)).mkdir(parents=True, exist_ok=True)
        plt.savefig("../Results/" + method + "/" + str(embed_size) + "/RocFold" + str(i + 1) + ".png")
        plt.close()
        y_pred = np.asarray([np.argmax(line) for line in y_pred])
        sen, spec = sen_and_spec(y_pred, y_test)
        fold_stats = {"Fold Num": (i + 1)/100,
                      "Accuracy": accuracy_score(y_test, y_pred),
                      "Precision": precision_score(y_test, y_pred),
                      "Recall": recall_score(y_test, y_pred),
                      "F1 score": f1_score(y_test, y_pred),
                      "Sensitivity": sen,
                      "Specificity": spec,
                      "AUC": roc_auc
                      }
        fold_stats = {k: round(v*100, 3) for k, v in fold_stats.items()}
        fold_stats.pop("Recall", None)
        ordered_keys = ["Fold Num", "Accuracy", "F1 score", "Precision", "Sensitivity", "Specificity", "AUC"]
        fold_stats = {k: fold_stats[k] for k in ordered_keys}
        for k, v in fold_stats.items():
            print(k + ":", v)
        buffer = fold_stats.copy()
        buffer["Embedding"] = embed_size
        output.append(buffer)
        for k in stats:
            stats[k].append(fold_stats[k])
        print("----------------------------------------------------------------------------------------------")
        print()
        i += 1

    print("Overall:")
    for k, v in stats.items():
        print(k + ":", np.mean(v))
    buffer = {k: np.mean(v) for k, v in stats.items()}
    buffer["Embedding"] = embed_size
    buffer["Fold Num"] = 1000
    output.append(buffer)
    o = pd.DataFrame(output)
    o.to_excel("../Results/" + method + ".xlsx")
    print()
    print("###############################################################################")
    print()

    x_auc.append(embed_size)
    y_auc.append(np.mean(stats["AUC"]))

step_size = 10
plt.plot(x_auc, y_auc, marker='o', color="orange")
plt.title("XGboost")
for i_x, i_y in zip(x_auc, y_auc):
    plt.text(i_x, i_y, '{:.2f}'.format(i_y))
plt.xticks(np.arange(min(x_auc), max(x_auc) + 5, step_size))
plt.xlabel("# of features")
plt.ylabel("Mean AUC for 5 folds")
pathlib.Path("../Results/" + "FeatureSizes").mkdir(parents=True, exist_ok=True)

import csv
with open("../Results/" + "FeatureSizes/" + method + ".csv", "w") as the_file:
    csv.register_dialect("custom", delimiter=" ", skipinitialspace=True)
    writer = csv.writer(the_file, dialect="custom")
    for tup in list(zip(x_auc, y_auc)):
        writer.writerow(tup)


sys.stdout.close()
