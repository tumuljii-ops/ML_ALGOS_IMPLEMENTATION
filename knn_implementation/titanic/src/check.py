import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")

print("train shape is:", train.shape)
print("test shape is:", test.shape)

# separate target
Y = train["Survived"]

# dropping target feature
X = train.drop("Survived", axis=1)

print("X shape:", X.shape)
print("Y shape is:", Y.shape)

print("null values:")
print(X.isnull().sum())

# drop columns
drop_cols = ["PassengerId", "Name", "Cabin", "Ticket"]

# drop the training features
X = X.drop(columns=drop_cols)

# drop from test set also
test_x = test.drop(columns=drop_cols)

print("after dropping\n", X.shape)
print("After dropping:\n", test_x.shape)

print("remaining col in X:\n")
print(X.columns)

# fill age with median
age_median = X["Age"].median()
X["Age"] = X["Age"].fillna(age_median)
test_x["Age"] = test_x["Age"].fillna(age_median)

# filling missing embarked values
embarked_mode = X["Embarked"].mode()[0]
X["Embarked"] = X["Embarked"].fillna(embarked_mode)
test_x["Embarked"] = test_x["Embarked"].fillna(embarked_mode)

# fill missing fare
fare_median = X["Fare"].median()
test_x["Fare"] = test_x["Fare"].fillna(fare_median)

print("missing values are:")
print(X.isnull().sum())
print(test_x.isnull().sum())

# converting categorical variables into numbers

# Encoding sex column
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
test_x["Sex"] = test_x["Sex"].map({"male": 0, "female": 1})

# one hot encoding Embarked
X = pd.get_dummies(X, columns=["Embarked"])
test_x = pd.get_dummies(test_x, columns=["Embarked"])

# align test columns with train columns
X, test_x = X.align(test_x, join="left", axis=1, fill_value=0)

print("after encoding and alignment:")
print("X shape is:", X.shape)
print("test_x shape is:", test_x.shape)
print("X columns are:")
print(X.columns)

# Standardization (Scaling)

# convert pandas DataFrame to numpy arrays
X_np = X.to_numpy(dtype=float)
test_np = test_x.to_numpy(dtype=float)

# computing mean and standard deviation of our arrays
mean = X_np.mean(axis=0)
std = X_np.std(axis=0)

# avoiding division by 0
std[std == 0] = 1

# Apply standardization formula
X_scaled = (X_np - mean) / std
test_scaled = (test_np - mean) / std

print("Scaling completed.")
print("Shape after scaling:", X_scaled.shape)

# splitting data set
X_train, X_val, Y_train, Y_val = train_test_split(
    X_scaled,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

print("TRAIN SHAPE IS:", X_train.shape)
print("VAL SHAPE IS:", X_val.shape)
print("Y_train shape is:", Y_train.shape)
print("Y_val shape is:", Y_val.shape)

# ---------------- KNN ----------------

# euclidean distance
def euclidean_dis(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predictions(X_train, X_val, Y_train, k):
    predictions = []

    # go through each validation passenger
    for val_points in X_val:
        dist_list = []

        for i in range(len(X_train)):
            dist = euclidean_dis(val_points, X_train[i])
            dist_list.append([dist, Y_train.iloc[i]])  # store distance and label

        # sort the distances
        dist_list.sort(key=lambda x: x[0])

        # take top k items
        top_k = dist_list[:k]

        # count votes
        one = 0
        zero = 0

        for item in top_k:
            label = item[1]

            if label == 1:
                one = one + 1
            else:
                zero = zero + 1

        if zero > one:
            predictions.append(0)
        else:
            predictions.append(1)

    return np.array(predictions)

# calling knn function
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

# predicting for test.csv now
test_pred = knn_predictions(X_scaled, test_scaled, Y, k=18)

print("prediction shape is:", test_pred.shape)
print("first 10 predictions are:", test_pred[:10])

#--------------------knn completed--------------------------------------

# ---------------- LOGISTIC REGRESSION ----------------

# ---------------- LOGISTIC REGRESSION ----------------

# initializing weights and bias
n_features = X_train.shape[1]

weights = np.zeros(n_features)
bias = 0.0

print("number of features are:", n_features)
print("weight shape is:", weights.shape)
print("bias initial value is:", bias)

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# prediction function
def predict_prob(X, weights, bias):
    z = np.dot(X, weights) + bias
    prob = sigmoid(z)
    return prob

train_probs = predict_prob(X_train, weights, bias)

print("10 predicted probabilities are:")
print(train_probs[:10])

print("Shape of train_probs is:", train_probs.shape)

# computing loss
def compute_loss(Y_true, Y_pred):
    eps = 1e-15

    Y_pred = np.clip(Y_pred, eps, 1 - eps)

    loss = -np.mean(
        Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred)
    )

    return loss

initial_loss = compute_loss(Y_train.to_numpy(), train_probs)

print("initial loss is:", initial_loss)

# computing gradients
def compute_gradient(X, Y_true, Y_pred):
    n_samples = X.shape[0]

    error = Y_pred - Y_true

    dw = (1 / n_samples) * np.dot(X.T, error)
    db = (1 / n_samples) * np.sum(error)

    return dw, db

dw, db = compute_gradient(X_train, Y_train.to_numpy(), train_probs)

print("shape of dw is:", dw.shape)
print("first 5 values of dw are:", dw[:5])
print("db is:", db)

# one manual update
lr = 0.01

weights = weights - lr * dw
bias = bias - lr * db

print("first 5 updated weights are:", weights[:5])
print("updated bias is:", bias)

updated_probs = predict_prob(X_train, weights, bias)

print("first 10 updated probabilities are:")
print(updated_probs[:10])

updated_loss = compute_loss(Y_train.to_numpy(), updated_probs)

print("loss after one update is:", updated_loss)

# training loop
def train_logistic_regression(X, Y, lr, epochs):
    n_features = X.shape[1]

    weights = np.zeros(n_features)
    bias = 0.0

    for epoch in range(epochs):
        # forward pass
        Y_pred = predict_prob(X, weights, bias)

        # compute loss
        loss = compute_loss(Y.to_numpy(), Y_pred)

        # compute gradients
        dw, db = compute_gradient(X, Y.to_numpy(), Y_pred)

        # update parameters
        weights = weights - lr * dw
        bias = bias - lr * db

        if epoch % 100 == 0:
            print("epoch:", epoch, "loss:", loss)

    return weights, bias

lr = 0.01
epochs = 1000

weights, bias = train_logistic_regression(X_train, Y_train, lr, epochs)

print("training completed")
print("first 5 final weights are:", weights[:5])
print("final bias is:", bias)

# class prediction
def predict_class(X, weights, bias):
    probs = predict_prob(X, weights, bias)

    preds = []

    for p in probs:
        if p >= 0.5:
            preds.append(1)
        else:
            preds.append(0)

    return np.array(preds)

val_pred = predict_class(X_val, weights, bias)

print("first 20 validation predictions are:")
print(val_pred[:20])

val_accuracy = np.mean(val_pred == Y_val.to_numpy())

print("Validation Accuracy is:", val_accuracy)

# evaluation metrics
def evaluate_metrics(Y_true, Y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(Y_true)):
        if Y_true[i] == 1 and Y_pred[i] == 1:
            tp = tp + 1
        elif Y_true[i] == 0 and Y_pred[i] == 0:
            tn = tn + 1
        elif Y_true[i] == 0 and Y_pred[i] == 1:
            fp = fp + 1
        else:
            fn = fn + 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = (2 * precision * recall) / (precision + recall)

    return tp, tn, fp, fn, accuracy, precision, recall, f1_score

tp, tn, fp, fn, accuracy, precision, recall, f1_score = evaluate_metrics(
    Y_val.to_numpy(),
    val_pred
)

print("Confusion Matrix:")
print("TP =", tp, "FP =", fp)
print("FN =", fn, "TN =", tn)

print("Accuracy =", accuracy)
print("Precision =", precision)
print("Recall =", recall)
print("F1 Score =", f1_score)

# prediction on test data using logistic regression
test_logistic_pred = predict_class(test_scaled, weights, bias)

print("test prediction shape is:", test_logistic_pred.shape)
print("first 10 logistic predictions are:", test_logistic_pred[:10])

# create submission file
submission_logistic = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_logistic_pred.astype(int)
})

submission_logistic.to_csv("submission_logistic.csv", index=False)

print("submission_logistic.csv created successfully")
print(submission_logistic.head())






  
     







