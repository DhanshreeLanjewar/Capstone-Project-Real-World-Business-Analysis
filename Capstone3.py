import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import ttest_ind

# Load dataset
df = pd.read_csv(r"C:\Users\dhans\Downloads\customer_churn (1).csv")


# DATA PREPROCESSING


# Convert TotalCharges to numeric (common issue)
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop rows with missing values
df.dropna(inplace=True)

# Convert categorical columns into numbers
df = pd.get_dummies(df, drop_first=True)


# SPLIT DATA

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# LOGISTIC REGRESSION

lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("🔹 Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# DECISION TREE

dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("\n🔹 Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))


# HYPOTHESIS TESTING
# Monthly Charges vs Churn


churned = df[df["Churn"] == 1]["MonthlyCharges"]
not_churned = df[df["Churn"] == 0]["MonthlyCharges"]

t_stat, p_val = ttest_ind(churned, not_churned)

print("\n📊 Hypothesis Testing (Monthly Charges)")
print("T-statistic:", round(t_stat, 3))
print("P-value:", round(p_val, 5))

if p_val < 0.05:
    print("Result: Significant difference (Reject H0)")
else:
    print("Result: No significant difference")