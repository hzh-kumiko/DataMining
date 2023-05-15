import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from dmba import regressionSummary, classificationSummary
from dmba import liftChart, gainsChart

car_df = pd.read_csv("ToyotaCorolla.csv")
exclude = ['price', 'model', 'color', 'fuel_type']
predictors = [s for s in car_df.columns if s not in exclude]
outcome = 'price'

train_x, valid_x, train_y, valid_y = train_test_split(car_df[predictors], car_df[outcome], test_size=0.4,
                                                      random_state=1)
model = LinearRegression()
model.fit(train_x, train_y)

train_pred = model.predict(train_x)
train_results = pd.DataFrame({'TOTAL_VALUE': train_y, 'prediction': train_pred, 'residual': train_y - train_pred})
# print(train_results)
regressionSummary(train_y, train_pred)
regressionSummary(valid_y, model.predict(valid_x))
pred_error_train = pd.DataFrame({'residual': train_y - train_pred, 'data_set': 'traing'})
print(pred_error_train, len(car_df))
pred_error_valid = pd.DataFrame({'residual': valid_y - model.predict(valid_x), 'data_set': 'validation'})
print(pred_error_valid)
boxdata_df = pred_error_train.append(pred_error_valid, ignore_index=True)
print(boxdata_df)

pred_v = pd.Series(model.predict(valid_x)).sort_values(ascending=False)
print(pred_v)
print(sum(pred_v[:57]))
fig,axes = plt.subplots(nrows=1,ncols=2)
ax = gainsChart(pred_v,ax=axes[0])
ax = liftChart(pred_v,ax=axes[1],labelBars=False)
plt.show()