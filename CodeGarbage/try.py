import pandas as pd
import matplotlib.pyplot as plt

fpa_roc = pd.read_csv("../Results/allROCs.csv")
fpa_roc = fpa_roc[["'False Positive Rate'", "'True Positive Rate'"]]
plt.plot(fpa_roc["'False Positive Rate'"], fpa_roc["'True Positive Rate'"])
plt.show()
