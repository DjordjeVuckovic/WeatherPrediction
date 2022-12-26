import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def evaluation_classif(conf_mat):
    TP = conf_mat[1, 1]
    TN = conf_mat[0, 0]
    FP = conf_mat[0, 1]
    FN = conf_mat[1, 0]
    precision = TP/(TP+FP)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    F_score = 2*precision*sensitivity/(precision+sensitivity)
    print('precision: ', precision)
    print('accuracy: ', accuracy)
    print('sensitivity/recall: ', sensitivity)
    print('specificity: ', specificity)
    print('F score: ', F_score)

# Load the data
df = pd.read_csv("dataset/final_dataset.csv")

# Assign labels to the samples
df["PM"] = df["PM_US Post"].apply(lambda x: "safe" if x <= 55.4 else ("unsafe" if x <= 150.4 else "dangerous"))
df=df.drop(['PM_US Post'], axis=1)
print(df.head())
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print('Samples num safe: ', sum(y=='safe'))
print('Samples num unsafe: ', sum(y=='unsafe'))
print('Samples num dangerous: ', sum(y=='dangerous'))

# Split the data into a training set and a test set, with 15% of the data reserved for testing
X = df.drop("PM", axis=1)
y = df["PM"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Define the cross-validation procedure using the KFold class
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Create a kNN classifier
model = KNeighborsClassifier()
# Calculating error for K values between 1 and 10
for m in ['hamming', 'euclidean']:
    error = []
    for i in range(1, 10):
        knn = KNeighborsClassifier(n_neighbors=i, metric=m)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(accuracy_score(y_test, pred_i))
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate for ' + m)
    plt.xlabel('K Value')
    plt.ylabel('Acc')
# Use cross-validation to determine the optimal number of neighbors
scores = []
for k in range(1, 11):
    model.n_neighbors = k
    score = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
    scores.append(score.mean())
# Select the optimal number of neighbors
optimal_k = scores.index(max(scores)) + 1
print("Optimal  number of neighbors:",optimal_k)

# Train the model with the optimal number of neighbors on the full training set
model.n_neighbors = optimal_k
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
print(optimal_k)
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
evaluation_classif(conf_mat)