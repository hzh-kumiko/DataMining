import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

cereals_df = pd.read_csv("Cereals.csv")
corr = cereals_df.corr().round(2)
cov = cereals_df.cov()
print(corr)
# sns.heatmap(corr, xticklabels=cov.columns, yticklabels=cov.columns, annot=True, fmt=".1f")
# plt.show()
plt.scatter(cereals_df.calories, cereals_df.rating)
# plt.show()
pcs = PCA(n_components=2)
print(cereals_df.describe())
pcs.fit(cereals_df[['calories', 'rating']])
pcsSummary = pd.DataFrame({'Standard deviation': np.sqrt(pcs.explained_variance_),
                           'Proportion of variance': pcs.explained_variance_ratio_,
                           'Cumulative proportion': np.cumsum(pcs.explained_variance_ratio_)}, index=['PCA1', 'PCA2'])
pcsSummary = pcsSummary.transpose()
# pcsSummary.columns = ['PCA1', 'PCA2']
# print(pcsSummary)
pcsComponents_df = pd.DataFrame(pcs.components_.transpose(), columns=['PC1', 'PC2'], index=['calories', 'rating'])

scores = pd.DataFrame(pcs.transform(cereals_df[['calories', 'rating']]), columns=['PC1', 'PC2'])
# print(scores)

