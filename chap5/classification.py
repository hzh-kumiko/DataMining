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

df = pd.read_csv("liftExample.csv")
cutoffs = [i * 0.1 for i in range(0, 11)]
accT = []
for cutoff in cutoffs:
    predicted = [1 if p > cutoff else 0 for p in df.prob]
    accT.append(accuracy_score(df.actual, predicted))
    classificationSummary(df.actual, predicted, class_names=['0', '1'])
    # print(classificationSummary)
# line_accuracy = plt.plot(cutoffs, accT, '-', label='Accuracy')
# line_error = plt.plot(cutoffs, [1 - acc for acc in accT], '--', label='error')
# plt.legend()
# plt.show()
fpr, tpr, _ = roc_curve(df.actual, df.prob)
roc_auc = auc(fpr, tpr)
print(fpr, tpr)

# df = pd.Series(df.prob).sort_values()  #带索引排序
df = df.sort_values(['prob'], ascending=False)
# liftChart(df.prob, labelBars=True)
print(df.actual)
gainsChart(df.actual*25-0.65)
plt.show()

