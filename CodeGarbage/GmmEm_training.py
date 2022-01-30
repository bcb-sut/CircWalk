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
from sklearn.mixture import GaussianMixture
import sys
method = "GaussianMixture"
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
for embed_size in range(10, 105, 10):
    print("Embedding Size:", embed_size)
    df_embeddings = pd.read_csv("../Data/entity_representations" + str(embed_size) + ".embeddings", header=None,
                                sep=" ")
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
        clf = GaussianMixture(n_components=2)
        clf.fit(train, y_train)
        y_pred = clf.predict(test)
        sen, spec = sen_and_spec(y_pred, y_test)

        # p = plot_roc_curve(clf, test, y_test)
        # pathlib.Path("../Results/" + method + "/" + str(embed_size)).mkdir(parents=True, exist_ok=True)
        # plt.savefig("../Results/" + method + "/" + str(embed_size) + "/RocFold" + str(i + 1) + ".png")
        # plt.close()

        fpr, tpr, thresh = roc_curve(y_test, clf.predict_proba(test)[:, 1])
        auc = roc_auc_score(y_test, y_pred)
        plt.plot(fpr, tpr, label='Fold %s ROC (area = %0.2f)' % (str(i + 1), auc))

        fold_stats = {"Fold Num": (i + 1) / 100,
                      "Accuracy": accuracy_score(y_test, y_pred),
                      "Precision": precision_score(y_test, y_pred),
                      "Recall": recall_score(y_test, y_pred),
                      "F1 score": f1_score(y_test, y_pred),
                      "Sensitivity": sen,
                      "Specificity": spec,
                      "AUC": 0}
        fold_stats = {k: round(v * 100, 3) for k, v in fold_stats.items()}
        for k, v in fold_stats.items():
            print(k + ":", v)
        for k in stats:
            stats[k].append(fold_stats[k])
        print("----------------------------------------------------------------------------------------------")
        print()
        i += 1
    print("Overall:")
    for k, v in stats.items():
        print(k + ":", np.mean(v))
    print()
    print("###############################################################################")
    print()

sys.stdout.close()
