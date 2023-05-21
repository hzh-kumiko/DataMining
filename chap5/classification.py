import matplotlib.pyplot as plt
import pandas as pd
from dmba import classificationSummary
from sklearn.metrics import accuracy_score, roc_curve, auc
from dmba import gainsChart, liftChart

owner_df = pd.read_csv("ownerExample.csv")

predicted = ['owner' if p > 0.5 else 'nonowner' for p in owner_df.Probability]
classificationSummary(owner_df.Class, predicted, class_names=['nonowner', 'owner'])
print(classificationSummary)

predicted = ['owner' if p > 0.25 else 'nonowner' for p in owner_df.Probability]
classificationSummary(owner_df.Class, predicted, class_names=['nonowner', 'owner'])
print(classificationSummary)

# 绘制准确度和临界值的关系
# 计算不同临界值下的准确率
df = pd.read_csv("liftExample.csv")
cutoffs = [i * 0.1 for i in range(0, 11)]
accT = []
for cutoff in cutoffs:
    predicted = [1 if p > cutoff else 0 for p in df.prob]
    accT.append(accuracy_score(df.actual, predicted))
    classificationSummary(df.actual, predicted, class_names=['0', '1'])
    # print(classificationSummary)
line_accuracy = plt.plot(cutoffs, accT, '-', label='Accuracy')
line_error = plt.plot(cutoffs, [1 - acc for acc in accT], '--', label='error')
plt.legend()
plt.show()

# TP FN
# FP TN
# 灵敏度 n11/(n11+n12)
# 特异度 n22/(m21+n22)
# ROC表示灵敏度，1-特异度值从临界值从1-0的变化
# AUC

fpr, tpr, _ = roc_curve(df.actual, df.prob)
roc_auc = auc(fpr, tpr)
print(fpr, tpr)

plt.figure(figsize=[5, 5])
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.legend(loc="lower right")
plt.show()
# df = pd.Series(df.prob).sort_values()  #带索引排序
df = df.sort_values(['prob'], ascending=False)
# liftChart(df.prob, labelBars=True)
print(df.actual)
