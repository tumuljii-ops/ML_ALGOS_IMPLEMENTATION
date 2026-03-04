import pandas as pd
train=pd.read_csv("../train.csv")
test=pd.read_csv("../test.csv")

print("training data is like:",train.shape)
print("testing data is like:",test.shape)

# print("first 5 rows:",train.head())

# print("data types of cols are:",train.info())

# print("missing values any ?:",train.isnull().sum())

# print("how data look mathematically:",train.describe())

#EDA STEPS
# print(train["SalePrice"].describe())

#visualisation of saleprice
import matplotlib.pyplot as plt

# plt.figure()
# plt.hist(train["SalePrice"], bins=30)
# plt.title("SalePrice Distribution")
# plt.xlabel("SalePrice")
# plt.ylabel("Count")
# plt.show() 


print("Skewness of salesprice:",train["SalePrice"].skew())

#applying log transformation to handle the skewness of the graph nature ,skew ness means heavy effect of outliers to our dataset

import numpy as np

train["SalePrice_log"] = np.log1p(train["SalePrice"])

print("Skewness after log:", train["SalePrice_log"].skew())

# plt.figure()
# plt.hist(train["SalePrice_log"], bins=30)
# plt.title("Log(SalePrice) Distribution")
# plt.show()

print("Total col:",train.shape[1])

num_cols=train.select_dtypes(include=[np.number]).columns #include numerical features
cat_cols=train.select_dtypes(exclude=[np.number]).columns #exlude numerical features

print("Length of numerical datas are:",len(num_cols))
print("Length of caterogical data:",len(cat_cols))

print("Top missing values are:")
print(train.isnull().sum().sort_values(ascending=False).head(15))

#percentage missing
percentage_miss=train.isnull().mean()

#cols with more than 80% missing data
drop_cols=percentage_miss[percentage_miss>0.8].index

print("Columns to drop are:")
print(drop_cols)

train=train.drop(columns=drop_cols)
test=test.drop(columns=drop_cols)
