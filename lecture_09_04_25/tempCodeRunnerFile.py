import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Create a dummy dataset
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

# Calculate L1 Loss (Mean Absolute Error)
l1_loss = mean_absolute_error(y_true, y_pred)
print(f"L1 Loss (Mean Absolute Error): {l1_loss}")

# Calculate L2 Loss (Mean Squared Error)
l2_loss = mean_squared_error(y_true, y_pred)
print(f"L2 Loss (Mean Squared Error): {l2_loss}")

# Plot the true and predicted values
plt.figure(figsize=(8, 5))
plt.plot(y_true, label='True Values', marker='o')
plt.plot(y_pred, label='Predicted Values', marker='x')
plt.title('True vs Predicted Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()