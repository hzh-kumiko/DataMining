import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from dmba import regressionSummary

housing_df = pd.read_csv('WestRoxbury.csv')
housing_df.columns = [s.strip().replace(' ', '_') for s in housing_df.columns]
# print(housing_df)
housing_df.REMODEL = housing_df.REMODEL.astype('category')
housing_df = pd.get_dummies(housing_df, prefix_sep='_', drop_first=True)
# print(housing_df)
# 目标值TOTAL_VALUE和无用信息TAX
excludeColumns = ['TOTAL_VALUE', 'TAX']
predictors = [s for s in housing_df.columns if s not in excludeColumns]
outcome = 'TOTAL_VALUE'

x = housing_df[predictors]
y = housing_df[outcome]

print(housing_df.columns)
# 训练集 验证集
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4, random_state=1)

model = LinearRegression()

model.fit(train_x, train_y)

train_pred = model.predict(train_x)
train_results = pd.DataFrame({'TOTAL_VALUE': train_y, 'prediction': train_pred, 'residual': train_y - train_pred})
print(train_results)
valid_pred = model.predict(valid_x)
valid_results = pd.DataFrame({'TOTAL_VALUE': valid_y, 'prediction': valid_pred, 'residual': valid_y - valid_pred})
print(valid_results)
print("score",model.score(train_x,train_y))
print(regressionSummary(train_results.TOTAL_VALUE, train_results.prediction))
print(regressionSummary(valid_results.TOTAL_VALUE, valid_results.prediction))

# 预测新数据的结果
new_data = pd.DataFrame(
    {'LOT_SQFT': [4200, 6444, 5035], 'YR_BUILT': [1960, 1940, 1925], 'GROSS_AREA': [2670, 2886, 3264],
     'LIVING_AREA': [1710, 1474, 1523], 'FLOORS': [2.0, 1.5, 1.9], 'ROOMS': [10, 6, 6], 'BEDROOMS': [4, 3, 2],
     'FULL_BATH': [1, 1, 1], 'HALF_BATH': [1, 1, 0], 'KITCHEN': [1, 1, 1], 'FIREPLACE': [1, 1, 0],
     'REMODEL_Old': [0, 0, 0], 'REMODEL_Recent': [0, 0, 1]})
new_pred = model.predict(new_data)
new_data.insert(new_data.shape[1],'prediction',new_pred)
print(new_data)