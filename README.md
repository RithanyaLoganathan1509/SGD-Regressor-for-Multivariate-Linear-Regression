# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and select input features (first three attributes) and output variables (house price and population).
2. Split the data into training and testing sets.
3. Apply Standard Scaling to both input features and output values.
4. Train a Multi-Output Regression model using SGDRegressor on the training data.
5. Predict the outputs, inverse transform the results, and evaluate performance using Mean Squared Error (MSE). 

## Program:
```
import numpy as np

X = np.array([
    [2, 80, 50],
    [3, 60, 40],
    [5, 90, 70],
    [7, 85, 80],
    [9, 95, 90]
], dtype=float)

y = np.array([50, 45, 70, 80, 95], dtype=float)

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

X = np.c_[np.ones(X.shape[0]), X]  # shape becomes (n_samples, n_features + 1)

n_features = X.shape[1]
weights = np.zeros(n_features)

# Hyperparameters
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    for i in range(X.shape[0]):
        xi = X[i]
        yi = y[i]
        y_pred = np.dot(xi, weights)
        error = y_pred - yi
        # Update weights
        weights -= learning_rate * error * xi

print("Trained Weights (including intercept):", weights)

y_pred_all = np.dot(X, weights)
print("Predicted values:", y_pred_all)

```

## Output:
<img width="911" height="75" alt="Screenshot 2026-02-11 134747" src="https://github.com/user-attachments/assets/1cffc554-9db9-47d3-ac22-0bf5df7d2f66" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
