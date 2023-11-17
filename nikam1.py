import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("D:/Programming/Python/temperatures.csv")

# Select features and target variable
x = df['JAN'].values.reshape(-1, 1)
y = df['FEB'].values.reshape(-1, 1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create and train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

plt.scatter(x_train, y_train, color='black')
plt.title('Linear Regression Training')
plt.xlabel('JAN Temperature')
plt.ylabel('FEB Temperature')
plt.show()

# Make predictions on the test set
y_pred = model.predict(x_test)

# Evaluate the model
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print("Mean Squared Error : ",mse)
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-Square: {r2}')

# Visualize the results
plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.title('Linear Regression Model')
plt.xlabel('JAN Temperature')
plt.ylabel('FEB Temperature')
plt.show()
