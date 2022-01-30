import pandas as pd
import json
import numpy as np
from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import xgboost as xgb
import pathlib

benchmark = pd.read_csv("../Data/clean_benchmark.csv")
benchmark = benchmark.values.tolist()
with open("../RawData/entityMap.json") as f:
    entity_map = json.load(f)
reversed_entity_map = {v: k for k, v in entity_map.items()}

embed_size = 10
df_embeddings = pd.read_csv("../Data/entity_representations" + str(embed_size) + ".embeddings", header=None, sep=" ")
df_embeddings.columns = ["id"] + ["val" + str(i) for i in range(1, df_embeddings.shape[1])]
df_embeddings['alle'] = [tuple(x) for x in
                         df_embeddings[["val" + str(i) for i in range(1, df_embeddings.shape[1])]].values.tolist()]
embeddings = dict(zip(df_embeddings.id, df_embeddings.alle))

disease_list = ["bladder cancer", "breast cancer", "cardiovascular disease", "cervical cancer", "colorectal cancer",
                "gastric cancer", "glioma", "leukemia", "lung cancer", "prostate cancer"]
disease_map = {dis: entity_map[dis] for dis in disease_list}
# print(disease_map)

x = []
y = []
circs_in_test = set()
train_pairs = []
for i, b in enumerate(benchmark):
    circ = embeddings[b[0]]
    dis = embeddings[b[1]]
    label = b[2]
    circs_in_test.add(b[0])

    x.append([circ[circiter] for circiter in range(len(circ))] + [dis[disiter] for disiter in range(len(dis))])
    y.append(label)
    train_pairs.append((b[0], b[1]))


x, y = shuffle(x, y, random_state=1024)
train = np.array(x)
y_train = np.array(y)

negatives = []
prune = int(len(train) * 2 / 6)
for i in range(len(train)):
    if y_train[i] == 0:
        negatives += [i]
        prune -= 1
    if prune <= 0:
        break

train = [train[i] for i in range(len(train)) if not i in negatives]
y_train = [y_train[i] for i in range(len(y_train)) if not i in negatives]
# clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(random_state=28, n_estimators=100), learning_rate=10**-2)
# clf.fit(train, y_train)
dtrain = xgb.DMatrix(np.array(train), label=np.array(y_train))
param = {
    'eta': 0.3,
    'max_depth': 7,
    'objective': 'multi:softprob',
    'num_class': 2}
steps = 50  # The number of training iterations
# dtest = xgb.DMatrix(method_specific_test, label=y_test)
evallist = [(dtrain, 'train')]
num_round = 64
bst = xgb.train(param, dtrain, num_round, evallist)
# y_pred = bst.predict(dtest, ntree_limit=bst.best_iteration + 1)
# from sklearn.svm import OneClassSVM
# clf = OneClassSVM(gamma='auto').fit(train)

# clf.predict(X)

circs_in_test = list(circs_in_test)
dis_in_test = list(disease_map.values())
test = []
k = 20
top_k_db = []
for j, d in enumerate(dis_in_test):
    dis = embeddings[dis_in_test[j]]
    predictions = []
    index = 0
    circ_dict = {}
    for i, c in enumerate(circs_in_test):
        circ = embeddings[circs_in_test[i]]
        test_sample = [circ[circiter] for circiter in range(len(circ))] + [dis[disiter] for disiter in range(len(dis))]
        if not ((circs_in_test[i], dis_in_test[j]) in train_pairs):
            dtest = xgb.DMatrix(np.array([test_sample]), label=None)
            prob = bst.predict(dtest, ntree_limit=bst.best_iteration + 1)[:,1]
            # prob = clf.predict_proba([test_sample])[:,1]
            predictions.append(prob[0])
            circ_dict[index] = circs_in_test[i]
            index += 1

    # top_k_indices = np.array(predictions).argsort()[-1 * min(index, k):][::-1]
    top_k_indices = [idx for idx in range(len(predictions)) if predictions[idx] > 0.5]
    for idx in list(top_k_indices):
        top_k_db.append([reversed_entity_map[dis_in_test[j]], reversed_entity_map[circ_dict[idx]], predictions[idx]])

df = pd.DataFrame(top_k_db,  columns=["Disease", "CircRNA", "Probability"])
df.to_excel("../Results/positive_cases.xlsx")