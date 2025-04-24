import numpy as np
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt

# Create a dummy dataset
X = np.array([[1], [2], [3], [4]])
y_true = np.array([3.0, -0.5, 2.0, 7.0])

# Range of lambda (alpha) values
lambda_values = np.logspace(-3, 3, 100)  # From 0.001 to 1000
ridge_coefficients = []
lasso_coefficients = []

# Train Ridge and Lasso regression for each lambda and store the coefficients
for lambda_value in lambda_values:
    ridge_model = Ridge(alpha=lambda_value)
    ridge_model.fit(X, y_true)
    ridge_coefficients.append(
        ridge_model.coef_[0]
    )  # Store the coefficient of the single feature

    lasso_model = Lasso(alpha=lambda_value)
    lasso_model.fit(X, y_true)
    lasso_coefficients.append(
        lasso_model.coef_[0]
    )  # Store the coefficient of the single feature

# Plot how the coefficients change with lambda
plt.figure(figsize=(8, 5))
plt.plot(lambda_values, ridge_coefficients, label="Ridge (L2)", marker="o")
plt.plot(lambda_values, lasso_coefficients, label="Lasso (L1)", marker="x")
plt.xscale("log")  # Use a logarithmic scale for lambda
plt.title("Coefficient vs Regularization Parameter (lambda)")
plt.xlabel("Lambda (Regularization Parameter)")
plt.ylabel("Coefficient Value")
plt.legend()
plt.grid(True)
plt.show()
