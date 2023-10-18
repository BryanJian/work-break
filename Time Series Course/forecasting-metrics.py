# Generate data points for the Laplace distribution
import os

# %%
import matplotlib.pyplot as plt
import numpy as np

# Define the parameters of the Laplace distribution
mu = 0  # Location parameter (peak)
b = 2  # Scale parameter (spread)

# Generate data points for the Laplace distribution
x = np.linspace(-10, 10, 1000)  # Values of x
pdf = (1 / (2 * b)) * np.exp(-abs(x - mu) / b)  # Laplace PDF

# Create a plot
plt.plot(x, pdf, label='Laplace PDF')
plt.xlabel('x')
plt.ylabel('PDF')
plt.title('Laplace Distribution')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Plot the above, but with varying mu and b using a for loop
mu = [-2, 0, 2]
b = [0.5, 1, 2]


x = np.linspace(-10, 10, 1000)  # Values of x

# Create a plot
for i in range(len(mu)):
    pdf = (1 / (2 * b[i])) * np.exp(-abs(x - mu[i]) / b[i])  # Laplace PDF
    plt.plot(x, pdf, label=f'mu={mu[i]}, b={b[i]}')
plt.xlabel('x')
plt.ylabel('PDF')
plt.title('Laplace Distribution')
plt.legend()
plt.grid(True)
plt.show()

# Saving this plot in an asset folder
if not os.path.exists('assets'):
    os.makedirs('assets')
plt.savefig('assets/laplace-distribution.png')

# %%
