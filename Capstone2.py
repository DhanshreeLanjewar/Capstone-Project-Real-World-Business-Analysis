import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\dhans\Downloads\customer_churn (1).csv")

# 1. Churn distribution
plt.figure()
df["Churn"].value_counts().plot(kind="bar")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.show()

# 2. Monthly Charges vs Churn
plt.figure()
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# 3. Tenure vs Churn
plt.figure()
sns.histplot(data=df, x="Tenure", hue="Churn", bins=30)
plt.title("Tenure Distribution by Churn")
plt.show()

# 4. Contract Type impact
contract_cols = [col for col in df.columns if "Contract_" in col]
df.groupby("Churn")[contract_cols].mean().T.plot(kind="bar")
plt.title("Contract Type vs Churn")
plt.show()

# 5. Correlation heatmap
plt.figure(figsize=(8,6))   # 👈 smaller figure

corr = df.corr()

sns.heatmap(
    corr,
    cmap="coolwarm",
    annot=False,
    linewidths=0.3,
    cbar=True
)

plt.title("Correlation Heatmap", fontsize=12)

# Smaller label fonts
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)

plt.tight_layout(pad=0.5)
plt.show()