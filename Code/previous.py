import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from sklearn.metrics import auc

methods = ["dmf", "ncpcda", "simccda", "circwalk", "gcncda2", "gcncda538"]
pretty_methods = {"dmf": "DMFCDA", "ncpcda": "NCPCDA", "simccda": "SIMCCDA", "circwalk": "Our Approach", "gcncda2":"GCNCDA (with GCN features)", "gcncda538": "GCNCDA (without GCN)"}
for fold in range(1, 6):
    for method in methods:
        s = "../Data/points/" + method + "-fold-" + str(fold) + ".xls"
        if method == "ncpcda":
            continue
        if method == "dmf":
            df = pd.read_excel(s, header=None)
            df = df.T
            df.columns = ["fpr", "tpr"]
            print(method)
            print(df)
        elif method == "circwalk":
            df = pd.read_excel(s)
            print(method)
            print(df)
        else:
            df = pd.read_excel(s, header=None, names=["fpr", "tpr"])
            print(method)
            print(df)
        roc_auc = auc(list(df["fpr"]), list(df["tpr"]))
        plt.plot(list(df["fpr"]), list(df["tpr"]), label='%s (AUC = %0.2f)' % (str(pretty_methods[method]), float(roc_auc * 100)))  # lw = 2

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="best")
    plt.title("ROC Comparison for Fold " + str(fold))
    pathlib.Path("../Results/" + "Previous").mkdir(parents=True, exist_ok=True)
    plt.savefig("../Results/" + "Previous" + "/Fold" + str(fold) + ".png")
    plt.close()
