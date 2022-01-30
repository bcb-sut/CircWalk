import pandas as pd
import matplotlib.pyplot as plt


methods = ["Adaboost", "xgboost", "SVM", "MultiLayerPerceptron", "Logistic", "RandomForest"]

pretty_methods = {"Adaboost": "ABRF", "xgboost": "XGB", "MultiLayerPerceptron": "MP",
                  "Logistic": "LR", "RandomForest": "RF", "SVM": "SVM"}
for method in methods:
    xy = pd.read_csv("../Results/FeatureSizes/" + method + ".csv", sep=" ", header=None)
    x = list(xy[xy.columns[0]])
    y = list(xy[xy.columns[1]])
    plt.plot(x, y, label=pretty_methods[method], marker="o")
    # for i_x, i_y in zip(x, y):
    #     plt.text(i_x, i_y, '{:.2f}'.format(i_y))

plt.legend(loc="best")
plt.xlabel("# of features")
plt.ylabel("Average AUC")
plt.savefig("../Results/FeatureSizes/FeatureComparison.png")
