import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

cereals_df = pd.read_csv("Cereals.csv")
# pcs = PCA()
pcs1 = PCA()
# pcs.fit(cereals_df.iloc[:, 3:].dropna(axis=0))

pcs1.fit(preprocessing.scale(cereals_df.iloc[:, 3:].dropna(axis=0)))

'''pcsSummary = pd.DataFrame({'Standard deviation': np.sqrt(pcs.explained_variance_),
                           'Proportion of variance': pcs.explained_variance_ratio_,
                           'Cumulative proportion': np.cumsum(pcs.explained_variance_ratio_)},
                          index=['PC{}'.format(i) for i in range(1, len(pcs.explained_variance_) + 1)])
pcsSummary = pcsSummary.transpose().round(4)'''
pcsSummary1 = pd.DataFrame({'Standard deviation': np.sqrt(pcs1.explained_variance_),
                            'Proportion of variance': pcs1.explained_variance_ratio_,
                            'Cumulative proportion': np.cumsum(pcs1.explained_variance_ratio_)},
                           index=['PC{}'.format(i) for i in range(1, len(pcs1.explained_variance_) + 1)])

pcsSummary1 = pcsSummary1.transpose().round(4)

pcsComponents_df = pd.DataFrame(pcs1.components_.transpose(), columns=pcsSummary1.columns, index=cereals_df.columns[3:])
# print(pcsComponents_df)
print(pcsComponents_df)
scores = pd.DataFrame(pcs1.transform(preprocessing.scale(cereals_df.iloc[:, 3:].dropna(axis=0))),
                      columns=['PC{}'.format(i) for i in range(1, len(pcs1.explained_variance_) + 1)])
print(scores)
