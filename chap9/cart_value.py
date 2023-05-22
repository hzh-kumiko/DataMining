import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
from dmba import plotDecisionTree, classificationSummary, regressionSummary

'''
与分类树对因变量进行分类相似，回归树用于对数值型因变量的预测，DecisionTreeRegressor，方法类似
设定候选参数
GridSearch搜索 得到最优参数
粗选后精选，在最优参数周围重新设置参数
best_estimator_得到最优模型
regressionSummary进行分析评估

不纯度指标 相对均值的差的平方和
'''
toyota_df = pd.read_csv('../chap5/ToyotaCorolla.csv')
toyota_df = toyota_df.rename(columns={'age_08_04': "age", 'quarterly_tax': 'tax'})
# print(toyota_df.columns)

predictors = ['age', 'km', 'fuel_type', 'hp', 'met_color', 'automatic', 'cc', 'doors', 'tax', 'weight']
outcome = 'price'

x = pd.get_dummies(toyota_df[predictors], drop_first=True)
y = toyota_df[outcome]

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4, random_state=1)

param_grid = {'max_depth': [5, 10, 15, 20, 25],
              'min_samples_split': [10, 20, 30, 40, 50],
              'min_impurity_decrease': [0, 0.001, 0.005, 0.01]}
gridSearch = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_x, train_y)
print('initial score', gridSearch.best_score_)
print('initial paras', gridSearch.best_params_)

param_grid = {'max_depth': list(range(2, 15)),
              'min_samples_split': list(range(5, 11)),
              'min_impurity_decrease': [s * 0.001 for s in range(11)]}
gridSearch = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_x, train_y)
print('improved score', gridSearch.best_score_)
print('improved paras', gridSearch.best_params_)

regTree = gridSearch.best_estimator_
regressionSummary(train_y, regTree.predict(train_x))
regressionSummary(valid_y, regTree.predict(valid_x))
