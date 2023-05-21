import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import matplotlib.pyplot as plt

# 计算两条记录的欧氏距离

mower_df = pd.read_csv("RidingMowers.csv")
print(mower_df.columns)
print(mower_df['Ownership'].value_counts())
# mower_df['Number'] = mower_df.index + 1
mower_df.insert(0, 'Number', mower_df.index + 1)  # 插在第一列后面,即为第二列
# print(mower_df)

trainData, validData = train_test_split(mower_df, test_size=0.4, random_state=26)

new_data = pd.DataFrame([{'Income': 60, 'Lot_Size': 20}])

owner = mower_df[mower_df['Ownership'] == 'Owner']
nonowner = mower_df[mower_df['Ownership'] == 'Nonowner']
plt.scatter(owner.Income, owner.Lot_Size, label='Owner', marker='o', color='C1')
plt.scatter(nonowner.Income, nonowner.Lot_Size, label='Nonowner', marker='D', color='C0')
plt.scatter(new_data.Income, new_data.Lot_Size, label='newData', marker='*', color='r')
for _, data in mower_df.iterrows():
    # print(data)
    plt.annotate(data.Number, xy=(data.Income, data.Lot_Size), xytext=(data.Income + 1, data.Lot_Size))
plt.legend()
# plt.show()

# 已有记录标准化

'''
1、fit

用于计算训练数据的均值和方差， 后面就会用均值和方差来转换训练数据

2、fit_transform

不仅计算训练数据的均值和方差，还会基于计算出来的均值和方差来转换训练数据，从而把数据转换成标准的正太分布

3、transform

很显然，它只是进行转换，只是把训练数据转换成标准的正态分布
'''

'''
1.归一化 preprocessing.MinMaxScalar, StandardScalar
2.上述的fit
3.transform转换
4.将预测扁蕾归一化，然后和其结果组合 concat
5.得到训练集和验证集，把要预测的数据也进行归一化
如果确定近邻个数
6.knn = NearestNeighbors(n_neighbors=3)，knn.fit(trainNorm.iloc[:, 0:2])，只需要输入变量
7.计算近邻距离和索引 distance, indices = knn.kneighbors(new_dataNorm)
如果想要确定最佳近邻个数
8.knn = KNeighborsClassifier(n_neighbors=k).fit(trainx, trainy)
9.计算在验证集上的准确度 accuracy_score(validy, knn.predict(validx))
10.采用准确率最高的k，重新用数据集的数据
allx = ownerNorm[['zIncome', 'zLot_Size']]
ally = ownerNorm.Ownership
knn = KNeighborsClassifier(n_neighbors=4).fit(allx, ally)
'''
scalar = preprocessing.MinMaxScaler()
scalar.fit(trainData[['Income', 'Lot_Size']])
pd1 = pd.DataFrame(scalar.transform(mower_df[['Income', 'Lot_Size']]), columns=['zIncome', 'zLot_Size'])
pd2 = mower_df[['Ownership', 'Number']]
ownerNorm = pd.concat([pd1, pd2], axis=1)
print(ownerNorm)
# 得到测试集和验证集
trainNorm = ownerNorm.iloc[trainData.index]
validNorm = ownerNorm.iloc[validData.index]
print(validNorm)

# 新纪录标准化
new_dataNorm = pd.DataFrame(scalar.transform(new_data), columns=['zIncome', 'zLot_Size'])

print(new_dataNorm)
# 设置近邻个数，用这个只填充了输入，没有输入和输出的关系，只计算最近点的距离
knn = NearestNeighbors(n_neighbors=3)
# 填充样本
knn.fit(trainNorm.iloc[:, 0:2])

# 计算新纪录的近邻
distance, indices = knn.kneighbors(new_dataNorm)
# print(distance, indices)
# print(trainNorm.iloc[indices[0], :])

# 计算验证集的近邻
distance, indices = knn.kneighbors(validNorm.iloc[:, 0:2])
# print(distance, indices)
# for i in range(len(indices)):
# print(trainNorm.iloc[indices[i], :])

# 选择不同k值来计算验证集的预测准确率
trainx = trainNorm[['zIncome', 'zLot_Size']]
trainy = trainNorm.Ownership
validx = validNorm[['zIncome', 'zLot_Size']]
validy = validNorm.Ownership

results = []
# 这里预测输入和输出的关系了
for k in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=k).fit(trainx, trainy)
    results.append({'k': k, 'accuracy': accuracy_score(validy, knn.predict(validx))})
    # results.append({'k': k, 'pre': knn.predict(validx)==validy, 'accuracy': accuracy_score(validy, knn.predict(validx))})
# print(pd.DataFrame(results))

# 上面的倒了k在4的时候准确率为0.9，重新训练所有的数据
# 计算距离和近邻索引
allx = ownerNorm[['zIncome', 'zLot_Size']]
ally = ownerNorm.Ownership
knn = KNeighborsClassifier(n_neighbors=4).fit(allx, ally)
distance, indices = knn.kneighbors(new_dataNorm)
# print(knn.predict(new_dataNorm))
# print('distance:', distance)
# print(indices)
print(ownerNorm.iloc[indices[0], :])

# 优点：  无参数法
# 缺点： 大型训练集下训练时间长，需要将为或搜索树加速搜索，训练集需要的记录数随变量个数呈指数级增长
