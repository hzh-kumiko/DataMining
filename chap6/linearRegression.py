import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, BayesianRidge
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

from dmba import regressionSummary, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score

car_df = pd.read_csv("ToyotaCorolla.csv")
car_df = car_df.rename(columns={'age_08_04': 'age'}).iloc[0:1000]
# print(car_df.columns)
predictions = ['age', 'km', 'fuel_type', 'hp', 'met_color',
               'automatic', 'cc', 'doors', 'quarterly_tax', 'weight']
outcome = 'price'
x = pd.get_dummies(car_df[predictions], drop_first=True)
y = car_df[outcome]

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4, random_state=1)
carReg = LinearRegression()
carReg.fit(train_x, train_y)
train_pred = carReg.predict(train_x)
car_pre = pd.DataFrame({'TOTAL_VALUE': train_y, 'prediction': train_pred, 'residual': train_y - train_pred})
print("car")
print(car_pre)
regressionSummary(train_y, train_pred)
coeff = pd.DataFrame({"coefficient": carReg.coef_}, index=x.columns)
print(coeff)

car_va_pre = carReg.predict(valid_x)
car_va = pd.DataFrame({'TOTAL_VALUE': valid_y, 'prediction': car_va_pre, 'residual': valid_y - car_va_pre})
print(car_va)
regressionSummary(valid_y, car_va_pre)
pred_error = pd.DataFrame({'residual': valid_y - car_va_pre, 'data set': 'training'})
pred_error.hist()
# plt.show()

######
'''穷举搜索法'''
# R2adj = 1 - (1-R2)(n-1)/(n-p-1)
# AIC, BIC


def train_model(variables):
    model = LinearRegression()
    model.fit(train_x[list(variables)],train_y)
    return model


def score_model(model, variables):
    pred_y = model.predict(train_x[list(variables)])
    return -adjusted_r2_score(train_y, pred_y, model)


allVariables = train_x.columns
results = exhaustive_search(allVariables,train_model,score_model)
print(results)
data=[]
for result in results:
    model = result['model']
    variables = list(result['variables'])
    AIC = AIC_score(train_y,model.predict(train_x[variables]),model)

