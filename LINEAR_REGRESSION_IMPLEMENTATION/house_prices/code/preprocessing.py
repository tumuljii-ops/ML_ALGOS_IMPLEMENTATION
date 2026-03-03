import pandas as pd
train=pd.read_csv("../train.csv")
test=pd.read_csv("../test.csv")

print("training data is like:",train.shape)
print("testing data is like:",test.shape)

# print("first 5 rows:",train.head())

# print("data types of cols are:",train.info())

# print("missing values any ?:",train.isnull().sum())

# print("how data look mathematically:",train.describe())