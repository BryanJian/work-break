# %%
## Plot as time series
import matplotlib.pyplot as plt
# %%
# Question 1: Using Numpy to generate 1000 samples from the standard normal distribution
import numpy as np

np.random.seed(123)
samples = np.random.normal(0, 1, 1000)
print(samples.shape)

x = np.arange(len(samples))
plt.scatter(np.arange(len(samples)), samples)  # Scatter plot
plt.show()

# %%
## Plot as histogram
plt.hist(samples, bins=50)
plt.show
# %%
# Question 2: Add a trend line to the noise
trend = np.linspace(0, 15, num=1000)
y = samples + trend
x = np.arange(len(y))
a, b = np.polyfit(x, y, 1)

plt.scatter(x, y)
plt.plot(x, a*x+b, color="red")
plt.show()

# %%
# Question 3: np.cumsum on noise
y_c = np.cumsum(samples)
plt.plot(np.arange(len(y_c)), y_c)
# This reminds me of a time series graph (e.g., stock performance)
# %%
# Question 4: Generate and plot 1000 samples from a multivariate normal
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]  # covariance is a measure of the joint variability of two random variables
x, y = np.random.multivariate_normal(mean, cov, 1000).T
plt.scatter(x, y)
plt.show()

# %%
# Question 5: Calculate the sample mean and covariance of the sample from Q4 without using np.cov and np.mean
# Mean
mean_y = np.sum(y) / len(y)
mean_x = np.sum(x) / len(x)
mean_res = [mean_x, mean_y]
mean_res

# Covariance matrix
cov_matrix = np.zeros([2, 2])

for i in range(len(x)):
    cov_matrix[0, 0] += (x[i] - mean_x) ** 2  # var x (diagonal element)
    cov_matrix[0, 1] += (x[i] - mean_x) * (y[i] - mean_y)  # covar x,y
    cov_matrix[1, 0] += (y[i] - mean_y) * (x[i] - mean_x)  # covar y,x (same as above)
    cov_matrix[1, 1] += (y[i] - mean_y) ** 2  # var y (diagonal element)

cov_matrix /= len(x)
cov_matrix

# %%
