import csv
import json

# with open("../RawData/entityMap.json", "r") as f:
#     d = json.load(f)
#
# df_benchmark = pd.read_csv("../RawData/benchmark.csv")
# df_benchmark["circrna"] = df_benchmark["circrna"].map(lambda x: d[x])
# df_benchmark["disease"] = df_benchmark["disease"].map(lambda x: d[x])
# benchmark = list(df_benchmark.to_records(index=None))
#
# with open("../Data/clean_benchmark.csv", "w") as f:
#     wr = csv.writer(f)
#     wr.writerows(benchmark)

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
import arff


benchmark = pd.read_csv("../Data/clean_benchmark.csv", header=None)
benchmark = benchmark.values.tolist()
for embed_size in range(10, 105, 10):
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
    y = np.array(y, dtype=int)
    yprim = np.array(list(map(lambda x: True if x == 1. else False, y)))

    weka_file = pd.DataFrame(x, columns=["Number" + str(i + 1) for i in range(len(x[0]))])
    weka_file["class"] = yprim
    # weka_file.to_csv("../Weka/" + str(embed_size) + ".csv", index=None)

    arff.dump("../Weka/" + str(embed_size) + '.arff'
              , weka_file.values
              , relation='relation_name'
              , names=weka_file.columns)
