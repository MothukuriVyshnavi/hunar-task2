# 🌦️ Weather Prediction using Linear Regression (with generated data)

# 📦 Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 🛠️ Generate a sample weather dataset and save as CSV
np.random.seed(42)
data = {
    'MinTemp': np.random.uniform(10, 25, 100),
    'MaxTemp': np.random.uniform(20, 40, 100),
    'Humidity': np.random.uniform(40, 90, 100)
}
df = pd.DataFrame(data)
df.to_csv("Weather_Data.csv", index=False)
print("✅ Sample Weather_Data.csv generated and loaded.")

# 📥 Load the dataset
df = pd.read_csv("Weather_Data.csv")

# 🔍 Show first few rows
print("📌 First 5 rows of data:")
print(df.head())

# 🔍 Check for missing values
print("\n🧹 Missing values in each column:")
print(df.isnull().sum())

# 🧹 Drop missing values (precaution)
df.dropna(inplace=True)

# 🎯 Define features and target
X = df[['MinTemp', 'Humidity']]  # Independent variables
y = df['MaxTemp']                # Target variable

# ✂️ Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 📈 Predict
y_pred = model.predict(X_test)

# 📊 Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# 📉 Plot Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k')
plt.xlabel("Actual MaxTemp")
plt.ylabel("Predicted MaxTemp")
plt.title("Actual vs Predicted MaxTemp")
plt.grid(True)
plt.show()