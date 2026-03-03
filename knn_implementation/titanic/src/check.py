import pandas as pd

train=pd.read_csv("../train.csv")
test=pd.read_csv("../test.csv")

print("train shape is:",train.shape)
print("test shape is:",test.shape)


# print(train.head)
# print(train.info)

# print("total missing values present are:")
# print(train.isnull().sum())

#separate target
Y=train["Survived"]

#dropping target feature
X=train.drop('Survived',axis=1)

print("X shape:",X.shape)
print("Y shape is:",Y.shape)

print("null values:",X.isnull().sum())

#drop columns
drop_cols=['PassengerId','Name','Cabin','Ticket']

#drop the training features
X=X.drop(columns=drop_cols)

#drop from test set also
test_x=test.drop(columns=drop_cols)

print("after dropping\n",X.shape)
print("After dropping:\n",test_x.shape)

print("remaining col in X:\n")
print(X.columns)


#fill age with median
age_median=X["Age"].median()
X["Age"].fillna(age_median,inplace=True)
test_x["Age"].fillna(age_median,inplace=True)

#filling missing embarked values
Embarked_mode=X["Embarked"].mode()[0]
X["Embarked"].fillna(Embarked_mode,inplace=True)
test_x["Embarked"].fillna(Embarked_mode,inplace=True)

#fill missing fare
fare_median=X["Fare"].median()
test_x["Fare"].fillna(fare_median,inplace=True)

print("missing values are:")
print(X.isnull().sum())
print(test_x.isnull().sum())

#converting categorical variables into numbers

#Encoding sex col
X["Sex"]=X["Sex"].map({"male":0,"female":1})
test_x["Sex"]=test_x["Sex"].map({"male":0,"female":1})

#make test_x coloums same as X coloumns 
test_x=test_x.reindex(columns=X.columns, fill_value=0)

print("after reindexing shapes are:")
print("X shape is:",X.shape)
print("test_x shape is:",test_x.shape)

#one hot encodeing embarked
X=pd.get_dummies(X,columns=["Embarked"])
test_x=pd.get_dummies(test_x,columns=["Embarked"])

print(X.shape)
print(test_x.shape)


#Standardization (Scaling)
import numpy as np

#convert pandas Dataframe to numpy arrays
X_np = X.to_numpy(dtype=float)
test_np = test_x.to_numpy(dtype=float)

#computing meand and standard deviation of our arrays
mean=X_np.mean(axis=0)
std=X_np.std(axis=0)

#avoiding division by 0
std[std==0]=1

# Apply standardization formula
X_scaled = (X_np - mean) / std
test_scaled = (test_np - mean) / std

print("Scaling completed.")
print("Shape after scaling:", X_scaled.shape)


#spliting data set

from sklearn.model_selection import train_test_split

X_train,X_val,Y_train,Y_val=train_test_split(
     X_scaled,
     Y,
     test_size=0.2,
     random_state=42,
     stratify=Y
    )

print("TRAIN SHAPE IS:",X_train.shape)
print("val_shape is:",X_val.shape)

#training the dataset

#euclidian distance 
def euclidian_dis(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))


def knn_predictions(X_train,X_val,Y_train,k):
    predictions=[]
    
    #go through each validation passenger
    
    for val_points in X_val:
        dist_list=[]
        for i in range(len(X_train)):
            dist=euclidian_dis(val_points,X_train[i])
            dist_list.append([dist,Y_train.iloc[i]]) #store dist as well as label also
            
        #sort the distances
        dist_list.sort(key=lambda x: x[0])
        
        #take top k items
        top_k=dist_list[:k]
        
        #count votes
        one=0
        zero=0
        
        for item in top_k:
            label=item[1]
            
            if(label==1):
                one=one+1
            else:
                zero=zero+1
                
        if(zero>one):
            predictions.append(0)
        else:
            predictions.append(1)
            
    return np.array(predictions)

#calling knn function

maxi = 0
best_k = 0

for k in range(1, 21):
    val_preds = knn_predictions(X_train, X_val, Y_train, k)
    accuracy = np.mean(val_preds == Y_val.to_numpy())

    if accuracy > maxi:
        maxi = accuracy
        best_k = k

print("Best K:", best_k)
print("Best Accuracy:", maxi)

#predicting for test.csv now

test_pred=knn_predictions(X_scaled,test_scaled,Y,k=18)

print("prediciton shape is:",test_pred.shape)
print("first 10 predictions are:",test_pred[:10])

#creating submission file
submission=pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_pred.astype(int)
}
)

submission.to_csv("submission.csv",index=False)
print("submission.csv created successfully")
print(submission.head())         
     
#“Start Linear Regression from scratch
# First do EDA + graphs (hist/box/scatter), then implement gradient descent. I want full intuition and step-by-step like Titanic.






