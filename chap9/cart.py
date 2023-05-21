import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
from dmba import plotDecisionTree, classificationSummary, regressionSummary

'''
1.Gini =  1 - sum(pk ^ 2), pk是类别k所占比例，最小是0，只有1个类别，最大1 - 1 / m, 等比例
2.信息熵 = -sum(pk * log(pk)) 0, 只有一个类别，log(m), m个类别等分
3.分割后，在每个分割后的矩阵中重新计算，并根据每个矩阵中样本数量加权求和，得到最小不纯度的分裂值
4.样本不同，分类情况会不同
5.会有过拟合情况

决策树分类器 classTree = DecisionTreeClassifier(random_state=0, max_depth=None)
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
 
为了避免过拟合 需要在适合的时候停止树的生长

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
classificationSummary(valid_y, fullClassTree.predict(valid_x))

# 交叉验证

treeClassifier = DecisionTreeClassifier(random_state=1)
scores = cross_val_score(treeClassifier, train_x, train_y, cv=5)
print(scores)

smallClassTree = DecisionTreeClassifier(max_depth=30, min_samples_split=20, min_impurity_decrease=0.01, random_state=1)
smallClassTree.fit(train_x, train_y)

tree.plot_tree(smallClassTree, feature_names=train_x.columns)
plt.show()

classificationSummary(train_y,smallClassTree.predict(train_x))
classificationSummary(valid_y,smallClassTree.predict(valid_x))
