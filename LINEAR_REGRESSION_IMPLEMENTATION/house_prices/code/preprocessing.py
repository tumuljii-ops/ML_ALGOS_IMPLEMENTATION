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

#filling missing values
#  recompute after dropping
num_cols = train.select_dtypes(include=[np.number]).columns
cat_cols = train.select_dtypes(exclude=[np.number]).columns

#filling numerical values with median
for col in num_cols:
    median_col=train[col].median()
    train[col]=train[col].fillna(median_col)
    if col in test.columns:
        test[col]=test[col].fillna(median_col)
        
        
#filling caterogical missing values with most frequent data

for col in cat_cols:
    mode_col=train[col].mode()[0]
    train[col]=train[col].fillna(mode_col)
    if col in test.columns:
        test[col]=test[col].fillna(mode_col)
        
print("Missing values after filling are:")
print(train.isnull().sum().sum())


#one hot encoding 
train=pd.get_dummies(train)
test=pd.get_dummies(test)

print("Train data set shape after encoding:",train.shape)
print("Testing data set shape after encoding:",test.shape)

#fixing col mismatch
train, test = train.align(test, join="left", axis=1, fill_value=0)

print("Train shape is now:",train.shape)
print("Test shape is now:",test.shape)

#separating target and feature

#target value
Y=train["SalePrice_log"]

#features are
X=train.drop(["SalePrice","SalePrice_log"],axis=1)
if "Id" in X.columns:
    X = X.drop(columns=["Id"])

print("Shape of X is:",X.shape)
print("Shape of target is:",Y.shape)

#spliting data set into training and validation

from sklearn.model_selection import train_test_split

X_train,X_val,Y_train,Y_val=train_test_split(
    X,Y,test_size=0.2,random_state=42
)

print("Training_set is:",X_train.shape)
print("Testing_shape is:",X_val.shape)

#now again doing feature scaling because some features have value 0 OR 1 ,AND some features have value 10000 or big integer values

import numpy as np
#convert to numpy array
X_train_np=X_train.to_numpy(dtype=np.float64)
X_val_np=X_val.to_numpy(dtype=np.float64)

#computing mean and std
mean=X_train_np.mean(axis=0)
std=X_train_np.std(axis=0)

#avoid dividing by zero
std[std==0]=1.0

#scale
X_train_scaled=(X_train_np - mean)/std
X_val_scaled=(X_val_np - mean)/std

print("Scaled train shape is:",X_train_scaled.shape)
print("Scaled val shape is:",X_val_scaled.shape)

#adding bias to our numpy array

n_train=X_train_scaled.shape[0]

n_val=X_val_scaled.shape[0]

#create col of 1
ones_train=np.ones((n_train,1))
ones_val=np.ones((n_val,1))

#attach to final dataset
X_train_final=np.hstack((ones_train,X_train_scaled))
X_val_final=np.hstack((ones_val,X_val_scaled))

print("shap is now:",X_train_final.shape)
print("val shape is now:",X_val_final.shape)

Y_train_np = Y_train.to_numpy(dtype=np.float64)
Y_val_np   = Y_val.to_numpy(dtype=np.float64)

#number of training samples and features 
n_sample=X_train_final.shape[0]
n_features=X_train_final.shape[1]

#initialize weights with 0
w=np.zeros(n_features)

#learning rate
lr=0.01

#number of epochs

epoch=2000

#ridge regularisation 
lam=0.1

for i in range (epoch):
    
    Y_pred=np.dot(X_train_final,w)
    
    #error
    error=Y_pred-Y_train_np
    
    #gradient
    grad=(2/n_sample)*np.dot(X_train_final.T,error)
    
    # add ridge penalty gradient (skip bias term w[0])
    grad[1:] = grad[1:] + (2.0 * lam * w[1:])
    
    #update weights
    w=w-lr*grad
    
    #print progress
    if i%200 ==0:
        mse=np.mean(error**2)
        print("epoch",i,"mse:",mse)
        
#predicting validation data
Y_val_pred=np.dot(X_val_final,w)

#error
error=Y_val_pred-Y_val_np

#square error
square=error*error

#mse
mse=np.mean(square)

#rmse
rmse=np.sqrt(mse)

print("validation mse:",mse)
print("validation rmse:",rmse)

#------using grid seach for hypermeter tuning----------

import numpy as np

def ridge_train_and_rmse(X_train, y_train, X_val, y_val, lam, lr, epochs):
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]

    w = np.zeros(n_features, dtype=np.float64)

    for _ in range(epochs):
        y_pred = np.dot(X_train, w)
        error = y_pred - y_train

        grad = (2.0 / n_samples) * np.dot(X_train.T, error)

        # ridge penalty (skip bias term w[0])
        grad[1:] += 2.0 * lam * w[1:]

        w = w - lr * grad

    # validation rmse
    val_pred = np.dot(X_val, w)
    val_mse = np.mean((val_pred - y_val) ** 2)
    val_rmse = np.sqrt(val_mse)

    return val_rmse, w

#---------------grid search----------------
lambdas = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]
lrs     = [0.001, 0.003, 0.01]
epochs_list = [1000, 2000, 4000]

best_rmse = float("inf")
best_params = None
best_w = None

for lam in lambdas:
    for lr in lrs:
        for epochs in epochs_list:
            rmse, w = ridge_train_and_rmse(
                X_train_final, Y_train_np,
                X_val_final, Y_val_np,
                lam, lr, epochs
            )

            print("lam:", lam, "lr:", lr, "epochs:", epochs, "rmse:", rmse)

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (lam, lr, epochs)
                best_w = w

print("\nBEST RMSE:", best_rmse)
print("BEST params (lam, lr, epochs):", best_params)

#---------final training + prediction +submission--------

import numpy as np
import pandas as pd

# combine train + validation
X_full = np.vstack((X_train_final, X_val_final))
Y_full = np.concatenate((Y_train_np, Y_val_np))

n_samples = X_full.shape[0]
n_features = X_full.shape[1]

# best hyperparameters
lam = 0.1
lr = 0.01
epochs = 4000

# initialize weights
w = np.zeros(n_features, dtype=np.float64)

# training loop (ridge regression)
for i in range(epochs):

    y_pred = np.dot(X_full, w)

    error = y_pred - Y_full

    grad = (2.0 / n_samples) * np.dot(X_full.T, error)

    # ridge penalty (skip bias)
    grad[1:] += 2 * lam * w[1:]

    w = w - lr * grad

# prepare test dataset (must match X columns exactly)
test_features = test[X.columns]
X_test = test_features.to_numpy(dtype=np.float64)

# scale using train statistics
X_test_scaled = (X_test - mean) / std

# add bias column
n_test = X_test_scaled.shape[0]
ones_test = np.ones((n_test, 1))
X_test_final = np.hstack((ones_test, X_test_scaled))

# prediction (log price)
Y_test_log_pred = np.dot(X_test_final, w)

# convert back to real price
Y_test_pred = np.expm1(Y_test_log_pred)

# create submission file
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": Y_test_pred
})

submission.to_csv("submission.csv", index=False)

print("Submission file created successfully!")







    
