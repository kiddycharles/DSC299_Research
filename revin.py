import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, t

# Set random seed for reproducibility
np.random.seed(42)

# Sample size
n = 50000000

# Distribution A: Standard Normal
dist_a = norm.rvs(size=n)
print(dist_a)

# Distribution B: Mixture of two normals
dist_b = np.concatenate([norm.rvs(-2, 0.5, size=n//2), norm.rvs(2, 0.5, size=n//2)])

# Ensure same mean and variance
dist_b = (dist_b - np.mean(dist_b)) / np.std(dist_b)
print(dist_b)

# Plot
plt.figure(figsize=(12, 6))
plt.hist(dist_a, bins=1000, alpha=0.5, label='Distribution A')
plt.hist(dist_b, bins=1000, alpha=0.5, label='Distribution B')
plt.legend()
plt.title('Two Distributions with Same Mean and Variance')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Print statistics
print(f"Mean A: {np.mean(dist_a):.4f}, Var A: {np.var(dist_a):.4f}")
print(f"Mean B: {np.mean(dist_b):.4f}, Var B: {np.var(dist_b):.4f}")

plt.close()

plt.figure(figsize=(12, 6))
plt.plot(dist_a, label='a')
plt.plot(dist_b, label='b')
plt.legend()
plt.title('Two time sequence with Same Mean and Variance')
plt.xlabel('Time')
plt.ylabel('Values')
plt.show()
