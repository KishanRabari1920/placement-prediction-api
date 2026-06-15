import pandas as pd
import pickle

# preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# metrics
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix)

# LOAD DATASET
df = pd.read_csv("../data/students.csv")
print("\nDataset Loaded Successfully")
print("\nFirst 5 Rows:")
print(df.head())

#DATASET INFORMATION
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# remove duplicates
df.drop_duplicates(inplace=True)

# INPUT FEATURES AND TARGET
X = df.drop("placed", axis=1)
y = df["placed"]

print("\nFeatures:")
print(X.columns)

print("\nTarget Column:")
print("placed")

# CLASS DISTRIBUTION
print("\nClass Distribution:")
print(y.value_counts())

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Test Split Completed")

print("\nX_train Shape:", X_train.shape)
print("X_test Shape :", X_test.shape)

# FEATURE SCALING
print("\nApplying StandardScaler...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaling Completed")

# LOGISTIC REGRESSION
print("LOGISTIC REGRESSION MODEL")

logistic_model = LogisticRegression()

# training
logistic_model.fit(X_train_scaled, y_train)

# prediction
logistic_predictions = logistic_model.predict(X_test_scaled)

# accuracy
logistic_train_accuracy = logistic_model.score(
    X_train_scaled,
    y_train
)

logistic_test_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

print("\nTrain Accuracy:")
print(logistic_train_accuracy)

print("\nTest Accuracy:")
print(logistic_test_accuracy)

# classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        logistic_predictions
    )
)

# confusion matrix
print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        logistic_predictions
    )
)

# RANDOM FOREST
print("RANDOM FOREST MODEL")

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# training
random_forest_model.fit(X_train, y_train)

# prediction
random_forest_predictions = random_forest_model.predict(X_test)

# accuracy
random_forest_train_accuracy = random_forest_model.score(
    X_train,
    y_train
)

random_forest_test_accuracy = accuracy_score(
    y_test,
    random_forest_predictions
)

print("\nTrain Accuracy:")
print(random_forest_train_accuracy)

print("\nTest Accuracy:")
print(random_forest_test_accuracy)

# classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        random_forest_predictions
    )
)

# confusion matrix
print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        random_forest_predictions
    )
)

# SUPPORT VECTOR MACHINE
print("SUPPORT VECTOR MACHINE MODEL")

svm_model = SVC()

# training
svm_model.fit(X_train_scaled, y_train)

# prediction
svm_predictions = svm_model.predict(X_test_scaled)

# accuracy
svm_train_accuracy = svm_model.score(
    X_train_scaled,
    y_train
)

svm_test_accuracy = accuracy_score(
    y_test,
    svm_predictions
)

print("\nTrain Accuracy:")
print(svm_train_accuracy)

print("\nTest Accuracy:")
print(svm_test_accuracy)

# classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        svm_predictions
    )
)

# confusion matrix
print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        svm_predictions
    )
)

# BEST MODEL SELECTION

print("MODEL COMPARISON")

print("\nLogistic Regression Accuracy:")
print(logistic_test_accuracy)

print("\nRandom Forest Accuracy:")
print(random_forest_test_accuracy)

print("\nSVM Accuracy:")
print(svm_test_accuracy)

best_accuracy = max(
    logistic_test_accuracy,
    random_forest_test_accuracy,
    svm_test_accuracy
)

# SAVE BEST MODEL

if best_accuracy == logistic_test_accuracy:
    best_model = logistic_model
    best_model_name = "Logistic Regression"
elif best_accuracy == random_forest_test_accuracy:
    best_model = random_forest_model
    best_model_name = "Random Forest"
else:
    best_model = svm_model
    best_model_name = "SVM"

# save model
with open("placement_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\nBest Model Saved Successfully")

print("\nBest Model:")
print(best_model_name)

print("\nBest Accuracy:")
print(best_accuracy)