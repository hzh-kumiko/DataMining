import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

housing_df = pd.read_csv('WestRoxbury.csv')
print(housing_df.head())
housing_df.rename(columns={'TOTAL VALUE ': 'TOTAL_VALUE'}, inplace=True)
housing_df.columns = [s.strip().replace(' ', '_') for s in housing_df.columns]
# print(housing_df)

# print(housing_df.loc[1:3])  # a-b
# print(housing_df.iloc[0:3])  # 0-2 [a:b] a——b-1

# TOP 10
'''print(housing_df['TOTAL_VALUE'].iloc[0:10])
print(housing_df['TOTAL_VALUE'].loc[0:10])
print(housing_df.iloc[0:10]['TOTAL_VALUE'])
print(housing_df.iloc[0:10].TOTAL_VALUE)'''

# concat
# print(pd.concat([housing_df.iloc[2:3,0:2],housing_df.iloc[4:6,0:2]],axis=0))

# rint(housing_df.TOTAL_VALUE.mean())
# print(housing_df.describe())
# print(housing_df['TAX'][0:4])

# random sample
# print(housing_df.sample(5))

# oversample
weights = [0.9 if rooms > 10 else 0.01 for rooms in housing_df.ROOMS]
# print(weights)
# print(housing_df.sample(5, weights=weights))
housing_df.REMODEL = housing_df.REMODEL.astype('category')
# print(housing_df.REMODEL.cat.categories)
# print(housing_df.REMODEL.dtype)

housing_df = pd.get_dummies(housing_df, prefix_sep='_', drop_first=True)
print(housing_df)
missingRows = housing_df.sample(10).index
print(missingRows)
housing_df.loc[missingRows, 'BEDROOMS'] = np.nan
print("number", housing_df['BEDROOMS'].count())
reduced_df = housing_df.dropna()
print(housing_df['BEDROOMS'].count())

medianBedrooms = housing_df.BEDROOMS.median()
print(medianBedrooms)
housing_df.BEDROOMS = housing_df.BEDROOMS.fillna(value=medianBedrooms)
print(housing_df.BEDROOMS.count())

df = housing_df.copy()

# normalizing

# calculate directly
norm_df = (housing_df - housing_df.mean()) / housing_df.std()
print(norm_df)
# calculate by API
scaler = preprocessing.StandardScaler()
print(housing_df.index)
norm_df = pd.DataFrame(scaler.fit_transform(housing_df), index=housing_df.index, columns=housing_df.columns)
# print("normalize", preprocessing.scale(housing_df))

print(norm_df)

# 数据重定标
norm_df = (housing_df - housing_df.min()) / (housing_df.max() - housing_df.min())

print(norm_df)

scaler = preprocessing.MinMaxScaler()
norm_df = pd.DataFrame(scaler.fit_transform(housing_df), index=housing_df.index, columns=housing_df.columns)
print(norm_df)

# 数据划分
# 60%训练集
# 40%验证集
trainData, validData = train_test_split(housing_df, test_size=0.4, random_state=1)
