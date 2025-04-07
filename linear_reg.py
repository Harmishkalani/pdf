import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample dataset
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Scores": [10, 20, 30, 40, 50, 60, 70, 80, 90]
}
df = pd.DataFrame(data)

# Splitting data
X = df[["Hours"]]
y = df["Scores"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.title('Hours vs Scores (Linear Regression)')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and show the plot
plt_path = "/mnt/data/linear_regression_plot.png"
plt.savefig(plt_path)
plt_path
