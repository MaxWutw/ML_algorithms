import numpy as np
import matplotlib.pyplot as plt
from K_means import KMeans
from sklearn.datasets import make_blobs

x, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=31105)
print(x.shape)

cluster = len(np.unique(y))

kmeans = KMeans(k=cluster, max_iters=100, plot_steps=False)
predict = kmeans.predict(x)

kmeans.plot()