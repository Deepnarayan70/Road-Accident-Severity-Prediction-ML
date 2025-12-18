import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix


#Load Dataset
df = pd.read_excel(r"C:\Users\DEEP NARAYAN\Documents\class\Predictive analysys\road accident dataset\RoadAccidentDataset.xlsx")

# EDA
print(df.shape)
print(df.info())
print("\nMissing values BEFORE imputation:")
print(df.isnull().sum())

cat_cols = df.select_dtypes(include="object").columns
imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = imputer.fit_transform(df[cat_cols])

print(df.isnull().sum())

df["Accident Date"] = pd.to_datetime(df["Accident Date"], dayfirst=True)
df["Year"] = df["Accident Date"].dt.year
df["Month"] = df["Accident Date"].dt.month
df["Day"] = df["Accident Date"].dt.day
df.drop("Accident Date", axis=1, inplace=True)


# OBJECTIVE 1: To analyze the distribution of casualty severity
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

y = df["Casualty Severity"]
X = df.drop("Casualty Severity", axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Casualty Severity Count:")
print(df["Casualty Severity"].value_counts())


plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Distribution of Casualty Severity")
plt.xlabel("Severity Level")
plt.ylabel("Count")
plt.show()

# OBJECTIVE 2: To study correlation among accident-related numeric features
plt.figure(figsize=(10,6))
sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm",
    annot=True,        # show numbers
    fmt=".2f"          # 2 decimal places
)
plt.title("Correlation Matrix of Accident Features")
plt.show()

# OBJECTIVE 3: To predict casualty severity using Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)

print("\nLogistic Regression Accuracy:", log_acc)

plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, log_pred), annot=True, fmt="d")
plt.title("Logistic Regression – Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# OBJECTIVE 4: To compare multiple classification algorithms

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

# Print Accuracy Scores
print("\nModel Accuracies:")
print("Logistic Regression:", log_acc)
print("Naive Bayes:", nb_acc)
print("KNN:", knn_acc)
print("Decision Tree:", dt_acc)

# Bar Chart of Accuracies
plt.figure(figsize=(6,4))
models = ["Logistic", "Naive Bayes", "KNN", "Decision Tree"]
accuracies = [log_acc, nb_acc, knn_acc, dt_acc]
plt.bar(models, accuracies)
plt.title("Classification Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()


# OBJECTIVE 5: Simplified Decision Tree for Interpretation

selected_features = [
    "Number of Vehicles",
    "Age of Casualty",
    "Road Surface",
    "Weather Conditions",
    "Lighting Conditions"
]

X_tree = df[selected_features]
y_tree = df["Casualty Severity"].astype(str)

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_tree, y_tree, test_size=0.2, random_state=42
)
dtree = DecisionTreeClassifier(max_depth=3, random_state=42)
dtree.fit(X_train_t, y_train_t)
y_pred_tree = dtree.predict(X_test_t)
acc_tree = accuracy_score(y_test_t, y_pred_tree)

print("Decision Tree Accuracy (Interpretation Model):", acc_tree)

plt.figure(figsize=(18,10))
plot_tree(
    dtree,
    feature_names=selected_features,
    class_names=dtree.classes_,
    filled=True
)
plt.title("Simplified Decision Tree for Accident Severity")
plt.show()

# OBJECTIVE 6: Feature Importance from Decision Tree

importance = dtree.feature_importances_

plt.figure(figsize=(6,4))
plt.barh(selected_features, importance)
plt.title("Important Factors Affecting Accident Severity")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()












