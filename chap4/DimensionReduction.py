import numpy as np
from scipy.optimize import minimize
import pandas as pd
import scipy
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pylab as plt
import seaborn as sns

housing_df = pd.read_csv('BostonHousing.csv')
corr = housing_df.corr().round(2)
print(corr)
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
            annot=True, fmt=".1f", vmin=-1, vmax=1, cmap="RdBu")
# plt.show()
analysis_data = pd.DataFrame({'mean:': housing_df.mean(), 'sd': housing_df.std(), 'min': housing_df.min()})
housing_df['RM_bin'] = pd.cut(housing_df.RM, range(0, 10), labels=False)
print(housing_df)
#print(housing_df.groupby(['RM_bin', 'CHAS']).MEDV.mean())
print(pd.pivot_table(housing_df, values='MEDV', index=['RM_bin'], columns=['CHAS'], aggfunc=[np.mean, np.std],
                     margins=True))
# housing_df.CAT_MEDV = housing_df.CAT_MEDV.astype('category')
# housing_df = pd.get_dummies(housing_df, prefix_sep='_', drop_first=True)
# print(housing_df.columns)

tb1 = pd.crosstab(housing_df.CAT_MEDV, housing_df.ZN)
print(tb1)
print(tb1.sum())
propTb1 = tb1/tb1.sum()
print(propTb1)
