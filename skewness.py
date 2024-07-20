import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Step 1: Generate Synthetic Data
np.random.seed(0)
n_samples = 1000
X = np.linspace(0, 10, n_samples)
noise = np.random.normal(0, 1, n_samples)
Y = 2 * X + 3 + noise + np.random.normal(0, 0.5, n_samples) ** 3  # Adding skewness and kurtosis

# Calculate true skewness and kurtosis
true_skewness = skew(Y)
true_kurtosis = kurtosis(Y)

# Step 2: Normalize the Data

# Standard Normalization (Mean and Variance)
Y_standard = (Y - np.mean(Y)) / np.std(Y)

# Extended Normalization (Mean, Variance, Skewness, and Kurtosis)
Y_centered = Y - np.mean(Y)
Y_standardized = Y_centered / np.std(Y)
Y_skew_adjusted = Y_standardized - true_skewness
Y_kurtosis_adjusted = (Y_skew_adjusted - true_kurtosis) / (1 + true_kurtosis)

# Step 3: Train a Simple Model

# Split data into training and test sets
X_train, X_test, Y_train_standard, Y_test_standard = train_test_split(X, Y_standard, test_size=0.3, random_state=0)
_, _, Y_train_kurtosis_adjusted, Y_test_kurtosis_adjusted = train_test_split(X, Y_kurtosis_adjusted,
                                                                             test_size=0.3, random_state=0)

# Standard Normalization Model
model_standard = LinearRegression()
model_standard.fit(X_train.reshape(-1, 1), Y_train_standard)
Y_pred_train_standard = model_standard.predict(X_train.reshape(-1, 1))
Y_pred_test_standard = model_standard.predict(X_test.reshape(-1, 1))

# Extended Normalization Model
model_kurtosis_adjusted = LinearRegression()
model_kurtosis_adjusted.fit(X_train.reshape(-1, 1), Y_train_kurtosis_adjusted)
Y_pred_train_kurtosis_adjusted = model_kurtosis_adjusted.predict(X_train.reshape(-1, 1))
Y_pred_test_kurtosis_adjusted = model_kurtosis_adjusted.predict(X_test.reshape(-1, 1))

# Step 4: Evaluate Overfitting

# Calculate Mean Squared Error
mse_train_standard = mean_squared_error(Y_train_standard, Y_pred_train_standard)
mse_test_standard = mean_squared_error(Y_test_standard, Y_pred_test_standard)
mse_train_kurtosis_adjusted = mean_squared_error(Y_train_kurtosis_adjusted, Y_pred_train_kurtosis_adjusted)
mse_test_kurtosis_adjusted = mean_squared_error(Y_test_kurtosis_adjusted, Y_pred_test_kurtosis_adjusted)

print("Standard Normalization - Train MSE:", mse_train_standard)
print("Standard Normalization - Test MSE:", mse_test_standard)
print("Extended Normalization - Train MSE:", mse_train_kurtosis_adjusted)
print("Extended Normalization - Test MSE:", mse_test_kurtosis_adjusted)
