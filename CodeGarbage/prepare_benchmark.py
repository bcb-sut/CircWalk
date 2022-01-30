import pandas as pd
import random

circ_dis = pd.read_csv("../RawData/dis-circ in mesh in circBasee.csv", header=None, sep="\t")
circ_dis.columns = ['dis', 'circ']
print(circ_dis.head())
print(len(circ_dis['dis'].unique()), len(circ_dis['circ'].unique()))
circ_dis.drop_duplicates()
circ_dis['label'] = 1
positives = list(circ_dis.to_records(index=False))
negatives = []
for circid in circ_dis['circ'].unique():
    edge_set = circ_dis[circ_dis['circ'] == circid]
    for disid in circ_dis['dis'].unique():
        if disid not in list(edge_set['dis']):
            negatives.append((disid, circid, 0))
dis_list = list(circ_dis['dis'])
circ_list = list(circ_dis['circ'])

print(positives)
print(negatives[:100])
print(len(negatives), len(circ_dis['dis'].unique()) * len(circ_dis['circ'].unique()) - len(positives))
negatives = random.sample(negatives, len(positives))
benchmark_tpl = positives + negatives

benchmark_tpl = [(i, j, k) for (j, i, k) in benchmark_tpl]
print(benchmark_tpl)
benchmark = pd.DataFrame(benchmark_tpl, columns=["circrna", "disease", "label"])
benchmark.to_csv("../RawData/benchmark.csv", index=None)
