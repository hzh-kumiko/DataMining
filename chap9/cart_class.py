import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
from dmba import plotDecisionTree, classificationSummary, regressionSummary

'''
不纯度指标
  1.Gini =  1 - sum(pk ^ 2), pk是类别k所占比例，最小是0，只有1个类别，最大1 - 1 / m, 等比例
  2.信息熵 = -sum(pk * log(pk)) 0, 只有一个类别，log(m), m个类别等分
3.分割后，在每个分割后的矩阵中重新计算，并根据每个矩阵中样本数量加权求和，得到最小不纯度的分裂值
4.样本不同，分类情况会不同
5.会有过拟合情况

决策树分类器 classTree = DecisionTreeClassifier(random_state=0, max_depth=None) 了解参数
获取混淆矩阵 classificationSummary

因为树结构不稳定，用交叉验证法评估 
  交叉验证，顾名思义，就是重复的使用数据，把得到的样本数据进行切分，组合为不同的训练集和测试集，用训练集来训练模型，
  用测试集来评估模型预测的好坏。在此基础上可以得到多组不同的训练集和测试集，某次训练集中的某样本在下次可能成为测试集中的样本，即所谓“交叉”。
 1、保留交叉验证 hand-out cross validation
 2、k折交叉验证 k-fold cross validation
    首先随机地将数据集切分为 k 个互不相交的大小相同的子集；
    然后将 k-1 个子集当成训练集训练模型，剩下的 (held out) 一个子集当测试集测试模型；
    将上一步对可能的 k 种选择重复进行 (每次挑一个不同的子集做测试集)；
    这样就训练了 k 个模型，每个模型都在相应的测试集上计算测试误差，得到了 k 个测试误差，对这 k 次的测试误差取平均便得到一个交叉验证误差
 3、留一交叉验证 leave-one-out cross validation
 
为了避免过拟合 需要在适合的时候停止树的生长，调节分类树的参数,
CHAID 
剪枝

如何对单棵树进行优化
采用随机森林或提升树的方法

总结树的优缺点
    1.常用于变量的选择，不需要变换变量，变量子集的选取是自动完成的
    2.会忽略预测变量之间的关系
    3。树对特性值的变化比较敏感
'''

owner_df = pd.read_csv('RidingMowers.csv')

classTree = DecisionTreeClassifier(random_state=0, max_depth=None)
classTree.fit(owner_df.drop(columns=['Ownership']), owner_df.Ownership)

print("Classes:{}".format(', '.join(classTree.classes_)))
# plotDecisionTree(classTree,feature_names=owner_df.columns[:2],class_names=classTree.classes_)
# tree.plot_tree(classTree,feature_names=owner_df.columns[:2],class_names=classTree.classes_)
# plt.axis('equal')

train, valid = train_test_split(owner_df, random_state=1, test_size=0.4)
classTree.fit(train.drop(columns=['Ownership']), train.Ownership)

print("Classes:{}".format(', '.join(classTree.classes_)))
# plotDecisionTree(classTree,feature_names=owner_df.columns[:2],class_names=classTree.classes_)
# tree.plot_tree(classTree, feature_names=train.columns[:2], class_names=classTree.classes_)
print("判断是否有割草机的准确率")
classificationSummary(valid.Ownership, classTree.predict(valid.drop(columns=['Ownership'])))
# plt.show()
# 上面两个结果很不同，与4所说相符

bank_df = pd.read_csv("UniversalBank.csv")
bank_df.drop(columns=['ID', 'ZIP Code'], inplace=True)

X = bank_df.drop(columns=['Personal Loan'])
Y = bank_df['Personal Loan']

train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=0.4, random_state=1)
fullClassTree = DecisionTreeClassifier(random_state=1)
fullClassTree.fit(train_x, train_y)
# tree.plot_tree(fullClassTree, feature_names=train_x.columns)

classificationSummary(train_y, fullClassTree.predict(train_x))
print('判断是否借贷准确率')
classificationSummary(valid_y, fullClassTree.predict(valid_x))

# 交叉验证
treeClassifier = DecisionTreeClassifier(random_state=1)
scores = cross_val_score(treeClassifier, train_x, train_y, cv=5)
print('用交叉验证计算其训练准确率', scores)

# 设定参数让树在一定时候停止生长
smallClassTree = DecisionTreeClassifier(max_depth=30, min_samples_split=20, min_impurity_decrease=0.01, random_state=1)
smallClassTree.fit(train_x, train_y)

tree.plot_tree(smallClassTree, feature_names=train_x.columns)
# plt.show()

classificationSummary(train_y, smallClassTree.predict(train_x))
print("停止生长")
classificationSummary(valid_y, smallClassTree.predict(valid_x))

# 调节分类树参数
param_grid = {'max_depth': [10, 20, 30, 40],
              'min_samples_split': [20, 40, 60, 80, 100],
              'min_impurity_decrease': [0, 0.0005, 0.001, 0.005, 0.01]}
gridSearch = GridSearchCV(DecisionTreeClassifier(random_state=1), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_x, train_y)
print("initial score: ", gridSearch.best_score_)
print("initial params", gridSearch.best_params_)

param_grid = {'max_depth': list(range(2, 16)),
              'min_samples_split': list(range(10, 22)),
              'min_impurity_decrease': [0.0001 * t for t in range(3, 12)]}
gridSearch = GridSearchCV(DecisionTreeClassifier(random_state=1), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_x, train_y)
print("improved score: ", gridSearch.best_score_)
print("improved params", gridSearch.best_params_)

'''
随机森林法
    每次挑选随机子集，拟合一颗分类树，就会有很多树
'''
rf = RandomForestClassifier(n_estimators=500, random_state=1)
rf.fit(train_x, train_y)

importance = rf.feature_importances_
# print(len(rf.estimators_))
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)  # 每列标准差

df = pd.DataFrame({'feature': train_x.columns, 'importance_': importance, 'std': std})
# print(sum(importance))
# print(df)

ax = df.plot(kind='barh', xerr='std', x='feature', legend=False)
ax.set_ylabel('')
# plt.show()
print("随机森林")
classificationSummary(valid_y, rf.predict(valid_x))

'''
提升树 
1.拟合一棵树
2.抽取样本，给误分类记录较大的选取概率
3.新样本上拟合一棵树
4.repeat
'''
boost = GradientBoostingClassifier()
boost.fit(train_x, train_y)
print("提升树")
classificationSummary(valid_y, boost.predict(valid_x))

