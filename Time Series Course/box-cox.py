# %%
!wget -nc https://lazyprogrammer.me/course_files/airline_passengers.csv

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import boxcox

# %%
df = pd.read_csv("airline_passengers.csv", index_col="Month", parse_dates=True)

df.head()
# %% [markdown]
# There's a long term **trend**, and short term **seasonality**
# and the amplitude of the seasonality increases over time

df["Passengers"].plot(figsize=(20, 8))

# %% [markdown]
# After applying a Power Transform (Square Root), we see that the trend
# has been squashed down slightly, but the seasonal amplitude is still increasing
# over time

df["SqrtPassengers"] = np.sqrt(df["Passengers"])
df["SqrtPassengers"].plot(figsize=(20, 8))
# %% [markdown]
# The Log Transform does a better job at squashing down the data to make
# it look more uniform in time
df["LogPassengers"] = np.log(df["Passengers"])
df["LogPassengers"].plot(figsize=(20, 8))


# %% [markdown]
data, lam = boxcox(df["Passengers"])
lam

# %%
df["BoxCoxPassengers"] = data
df["BoxCoxPassengers"].plot(figsize=(20, 8))
# %%
df["Passengers"].hist(bins=20)
# %%
df["SqrtPassengers"].hist(bins=20)
# %%
df["LogPassengers"].hist(bins=20)
# %%
df["BoxCoxPassengers"].hist(bins=20)
# %%
