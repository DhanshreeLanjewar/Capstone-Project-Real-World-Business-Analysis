import pandas as pd

# Load data
df = pd.read_csv(r"C:\Users\dhans\Downloads\customer_churn (1).csv")

# Drop CustomerID (not useful for analysis)
df = df.drop(columns=["CustomerID"])

# Check missing values
print(df.isnull().sum())

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Encode categorical variables
categorical_cols = ["Contract", "PaymentMethod", "PaperlessBilling"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Save cleaned data
df_encoded.to_csv("data/cleaned_data.csv", index=False)

print("Data cleaning completed.")