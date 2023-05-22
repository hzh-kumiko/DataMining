import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from dmba import classificationSummary, gainsChart

'''
分类预测变量 朴素贝叶斯
 P(C1|x1,..xn) = P(C1)[P(x1|C1)P(x2|C1)···P(xn|C1)] / P(C1)[P(x1|C1)···P(xn|C1)]+...P(Cm)[P(x1|Cm)···P(xn|Cm)]
'''

# 预测航班是否延误
# 只分析该数据集的五个变量 DAY_WEEK, CRS_DEP_TIME, ORIGIN, DEST, CARRIER
delay_df = pd.read_csv("FlightDelays.csv")

delay_df.DAY_WEEK = delay_df.DAY_WEEK.astype('category')
delay_df = delay_df.rename(columns={'Flight Status': 'Flight_Status'})
delay_df.Flight_Status = delay_df.Flight_Status.astype('category')
print(delay_df.columns)

delay_df.CRS_DEP_TIME = round(delay_df.CRS_DEP_TIME / 100)
delay_df.CRS_DEP_TIME = delay_df.CRS_DEP_TIME.astype('category')
# print(delay_df)

predictors = ['DAY_WEEK', 'CRS_DEP_TIME', 'ORIGIN', 'DEST', 'CARRIER']
outcome = 'Flight_Status'

# 把分类的预测变量拆分成各个列
X = pd.get_dummies(delay_df[predictors])
Y = delay_df['Flight_Status']
print(list(Y.cat.categories))
train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=0.4, random_state=1)

'''
1.分类变量get_dummies
2.朴素贝叶斯模型delay_nb = MultinomialNB(alpha=0.01)，delay_nb.fit(train_x, train_y)
3.预测结果概率delay_nb.predict_proba

另一种 用透视表计算每个种类航班延误与否的情况, 
1.train_df.Flight_Status.value_counts() / len(train_df)得到P(ontime)&P(delayed)概率
2.pivot_table 得到 这就得到了P(xi|C1/2)的概率
3.计算分子 P(C1)[P(x1|C1)P(x2|C1)···P(xn|C1)]

如果可以在数据集上找到相同记录的数据则直接看 
'''
delay_nb = MultinomialNB(alpha=0.01)
delay_nb.fit(train_x, train_y)

pred_train_prob = delay_nb.predict_proba(train_x)
pred_valid_prob = pd.DataFrame(delay_nb.predict_proba(valid_x), index=valid_y.index, columns=['delayed', 'ontime'])
y_valid_pred = delay_nb.predict(valid_x)

train_df, valid_df = train_test_split(delay_df, test_size=0.4, random_state=1)
# precision 保留小数位数
pd.set_option('precision', 4)

# 计算准时和延误的概率
print(train_df.Flight_Status.value_counts() / len(train_df))
for pre in predictors:
    df = train_df[['Flight_Status', pre]]
    pivot = df.pivot_table(index='Flight_Status', columns=pre, aggfunc=len)
    average_pivot = pivot.apply(lambda x: x / sum(x), axis=1)
    print(average_pivot)
    print()

pd.reset_option('precision')

# DL公司 从DCA到LGA的航班， 起飞时间10点 日期周日 计算朴素概率的分子：
# P(delayed|Carrier=DL,DAY_WEEK=7,DEP_TIME=10,DEST=LGA,ORIGIN=DCA) 分子为上面计算出来的各项对应的相乘乘以P(延误)即(0.2)


df = pd.concat([pd.DataFrame({'actual': valid_y, 'predicted': y_valid_pred}), pred_valid_prob], axis=1)
print(df)
# print(valid_x.columns)
# 在验证集上存在的化
mask = ((valid_x.CARRIER_DL == 1) & (valid_x.DAY_WEEK_7 == 1) & (valid_x.DEST_LGA == 1) & (
        valid_x['CRS_DEP_TIME_10.0'] == 1) & (valid_x.ORIGIN_DCA == 1))
print(mask)
print(df[mask])
valid = pd.DataFrame(y_valid_pred, index=valid_y.index, columns=['actualll'])
classificationSummary(train_y, delay_nb.predict(train_x), class_names=['delayed', 'ontime'])
classificationSummary(valid_y, y_valid_pred, class_names=['delayed', 'ontime'])
df = pd.concat([pd.DataFrame({'actual': 1 - valid_y.cat.codes}), pred_valid_prob.delayed], axis=1)
df = df.sort_values(by=['delayed'], ascending=False).reset_index(drop=True)
gainsChart(df.actual)
plt.show()
print(df)
