import yfinance as yf

# Fetch stock data
ticker = "AAPL"
start_date = "2010-01-01"
end_date = "2023-09-01"
data = yf.download(ticker, start=start_date, end=end_date)

import numpy as np

import pandas as pd

# Calculate log returns
data["Log Return"] = np.log(data["Close"] / data["Close"].shift(1))
data = data.dropna()  # Remove rows with missing values

import matplotlib.pyplot as plt

# Plot log returns
plt.figure(figsize=(10, 6))
plt.plot(data.index, data["Log Return"])
plt.xlabel("Date")
plt.ylabel("Log Return")
plt.title("Log Returns of AAPL Stock")
plt.grid(True)

plt.show()

import sklearn
from sklearn.cluster import KMeans

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data["Cluster"] = kmeans.fit_predict(data[["Log Return"]])

from sklearn.mixture import GaussianMixture

# Apply Gaussian Mixture Models
gmm = GaussianMixture(n_components=3, random_state=42)
data["Regime"] = gmm.fit_predict(data[["Log Return"]])

cluster_means = data.groupby("Cluster")["Log Return"].mean()
cluster_means.plot(kind="bar", figsize=(10, 6))
plt.xlabel("Cluster")
plt.ylabel("Average Log Return")
plt.title("Average Log Return for Each Cluster")
plt.grid(True)

plt.show()

import numpy as np

regime_probs = np.exp(gmm.score_samples(data[["Log Return"]]))
data["Regime Probability"] = regime_probs

regime_means = data.groupby("Regime")["Regime Probability"].mean()
regime_means.plot(kind="bar", figsize=(10, 6))
plt.xlabel("Regime")
plt.ylabel("Average Regime Probability")
plt.title("Average Regime Probability for Each Regime")
plt.grid(True)

plt.show()

