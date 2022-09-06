import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from Principle_Component_Analysis import PCA

data = datasets.load_iris()

x = data.data
# print(x.shape) # shape : (150, 4)
y = data.target
# print(y.shape)
pca = PCA(n_components=2) # instances
pca.fit(x)

x_projected = pca.transform(x)

print(x.shape)
print(x_projected.shape)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

plt.scatter(x1, x2, c=y, edgecolors='none', alpha=0.8)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar()
plt.show()