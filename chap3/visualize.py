import os
import calendar
import numpy as np
import networkx as nx
import pandas as pd
from pandas.plotting import scatter_matrix, parallel_coordinates
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pylab as plt

housing_df = pd.read_csv('BostonHousing.csv')
# print(housing_df)

Amtrak_df = pd.read_csv('Amtrak.csv', squeeze=True)
print(Amtrak_df.head())
Amtrak_df['Date'] = pd.to_datetime(Amtrak_df.Month, format='%d/%m/%Y')
print(Amtrak_df.head())

ridership_ts = pd.Series(Amtrak_df.Ridership.values, index=Amtrak_df.Date)
# print(ridership_ts)

# 前用pandas画图 后用matplotlib画图
# ridership_ts.plot(ylim=[1300,2300],legend=False)
# plt.plot(ridership_ts.index, ridership_ts)
# plt.xlabel('Year')
# plt.ylabel('Ridership')

# housing_df.plot.scatter(x='LSTAT', y='MEDV', legend=True)
# plt.scatter(housing_df.LSTAT, housing_df.MEDV, color='C2', facecolor='none')
# plt.xlabel('LSTAT')
# plt.ylabel('MEDV')
# plt.legend()
dataForPlot = housing_df.groupby('CHAS').mean()
'''ax = dataForPlot.MEDV.plot(kind='bar')
plt.xticks(rotation=0)
ax.set_ylabel('Avg.MEDV')
print(dataForPlot)
'''

# 条形图
# fig, axes = plt.subplots(nrows=1, ncols=4)
'''ax.bar(dataForPlot.index,dataForPlot.MEDV,color=['C5','C1'])
ax.set_xticks([0,1],False)
ax.set_xlabel('CHAS')
ax.set_ylabel('MEDV')
'''

# 直方图
# ax = housing_df.MEDV.hist()
'''ax.hist(housing_df.MEDV)
ax.set_xlabel('MEDV')
ax.set_ylabel('count')'''

'''ax.hist(housing_df.MEDV)
ax.set_axisbelow(True)
ax.grid(which='major',color='grey',linestyle='--')
ax.set_xlabel('MEDV')
ax.set_ylabel('count')'''

# 箱线图
'''housing_df.boxplot(column='NOX',by='CAT_MEDV',ax=axes[0])
plt.tight_layout()
plt.show()'''

# 热力图
'''fig, ax = plt.subplots()
corr = housing_df.corr()
print(corr)
sns.heatmap(corr, vmin=-1, vmax=1, cmap="RdBu")
# sns.heatmap(corr,annot=True,fmt=".1f",cmap="RdBu",center=0,ax=ax)
plt.show()'''

# housing_df.plot.scatter(x='LSTAT', y='NOX', color=['C0' if c == 1 else 'C1' for c in housing_df.CAT_MEDV])
#_, ax = plt.subplots()

# 散点图 空心圆 分类
'''for catValue, color in (0, 'C1'), (1, 'C0'):
    subset_df = housing_df[housing_df.CAT_MEDV == catValue]
    ax.scatter(subset_df.LSTAT, subset_df.NOX, color='none', edgecolor=color)
ax.set_xlabel('LSTAT')
ax.set_ylabel('NOX')
ax.legend(["CAT.MEDV 0", "CAT.MEDV 1"])'''

dataForPlot_df = housing_df.groupby(['CHAS', 'RAD']).mean()['MEDV']
ticks = set(housing_df.RAD)

for i in range(2):
    for t in ticks.difference(dataForPlot_df[i].index):
        dataForPlot_df.loc[(i, t)] = 0
# print(dataForPlot_df)
# print(dataForPlot_df)
'''dataForPlot_df = dataForPlot_df[sorted(dataForPlot_df.index)]
yRange = [0,max(dataForPlot_df)*1.1]
fig,axes = plt.subplots(nrows=2,ncols=1)
dataForPlot_df[0].plot.bar(x='RAD',ax=axes[0],ylim=yRange)
dataForPlot_df[1].plot.bar(x='RAD',ax=axes[1],ylim=yRange)
axes[0].annotate('CHAS = 0',xy=(3.5,45))
axes[1].annotate('CHAS = 1',xy=(3.5,45))'''
utilities_df = pd.read_csv('Utilities.csv')
ax = utilities_df.plot.scatter(x='Sales',y='Fuel_Cost', figsize=(6,6))
points = utilities_df[['Sales','Fuel_Cost','Company']]
_ = points.apply(lambda x: ax.text(*x,horizontalalignment='left',verticalalignment='bottom',fontsize=8),axis=1)

plt.show()
