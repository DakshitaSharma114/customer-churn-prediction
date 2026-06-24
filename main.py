import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("Data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ==========================
# DATA CLEANING
# ==========================
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df = df.dropna()

# ==========================
# TARGET COLUMN
# ==========================
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# ==========================
# DROP CUSTOMER ID
# ==========================
df = df.drop("customerID", axis=1)

# ==========================
# CONVERT TEXT TO NUMBERS
# ==========================
df = pd.get_dummies(df, drop_first=True)

# ==========================
# FEATURES & TARGET
# ==========================
X = df.drop("Churn", axis=1)
y = df["Churn"]

joblib.dump(X.columns.tolist(), "model_columns.pkl")
# ==========================
# TRAIN TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# LOGISTIC REGRESSION MODEL
# ==========================
model = LogisticRegression(max_iter=2000)

model.fit(X_train, y_train)

# ==========================
# PREDICTIONS
# ==========================
y_pred = model.predict(X_test)

# ==========================
# EVALUATION
# ==========================
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================
# TOP CHURN FACTORS
# ==========================
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

feature_importance = feature_importance.sort_values(
    by="Coefficient",
    ascending=False
)

print("\nTop 10 Features Increasing Churn:\n")
print(feature_importance.head(10))

print("\nTop 10 Features Reducing Churn:\n")
print(feature_importance.tail(10))
print("\n==========================")
print("CUSTOMER RISK ANALYSIS")
print("==========================")

probabilities = model.predict_proba(X_test)

for i in range(5):

    churn_probability = probabilities[i][1] * 100

    if churn_probability < 30:
        risk = "LOW 🟢"

    elif churn_probability < 70:
        risk = "MEDIUM 🟡"

    else:
        risk = "HIGH 🔴"

    print(
        f"\nCustomer {i+1}"
    )

    print(
        f"Churn Probability: {churn_probability:.2f}%"
    )

    print(
        f"Risk Level: {risk}"
    )
    joblib.dump(model, "churn_model.pkl")

print("\nModel Saved Successfully!")