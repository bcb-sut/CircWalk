import pandas as pd
import json
from collections import defaultdict

# merged = pd.read_csv("../RawData/MergedNetwork.edgelist", header=None, sep="\t")
# merged.columns = ["node1", "node2"]
# print(merged.head())
# print(merged.shape)
#
# entries = list(set(merged["node1"]).union(set(merged["node2"])))
# d = defaultdict(int)
# for i, e in enumerate(entries):
#     d[e] = i
#
# merged["node1"] = merged["node1"].map(lambda row: d[row])
# merged["node2"] = merged["node2"].map(lambda row: d[row])
# print(merged.head(50))
#
# merged.to_csv("../RawData/intMergedNetwork.edgelist", sep="\t", index_label=None, index=None)
# with open("../RawData/entityMap.json", "w") as f:
#     json.dump(d, f)
