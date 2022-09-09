## K means algorithm is a kind of unsupervised learning, and it's main idea is to cluster the data.
import numpy as np
import matplotlib.pyplot as plt
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:
    def __init__(self, k=3, max_iters=100, plot_steps=False):
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.k)]
        # mean feature vector for each cluster
        self.centroids = []

    def predict(self, x):
        self.x = x
        self.n_samples, self.n_features = x.shape

        # init centroids
        random_centroids = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.x[idx] for idx in random_centroids]
        # optimization
        for _ in range(self.max_iters):
            # clusters
            self.clusters = self.clustering(self.centroids)
            if self.plot_steps:
                self.plot()
            # update centroids
            old_centroids = self.centroids
            self.centroids = self.get_new_centroids(self.clusters)
            if self.plot_steps:
                self.plot()
            # check if converged
            if self.is_changed(old_centroids, self.centroids):
                break

        # return cluster labels
        return self.get_label(self.clusters)

    def clustering(self, centroids):
        cluster_sample = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.x):
            cent_idx = self.closest_centroid(sample, centroids)
            cluster_sample[cent_idx].append(idx)
        return cluster_sample

    def closest_centroid(self, sample, centroids):
        distance = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distance)
        return closest_idx

    def get_new_centroids(self, clusters):
        new_centroids = np.zeros((self.k, self.n_features))
        for idx, cluster_data in enumerate(clusters):
            # print(cluster_data)
            new_cluster_mean = np.mean(self.x[cluster_data], axis=0)
            new_centroids[idx] = new_cluster_mean
        return new_centroids

    def is_changed(self, old_centroids, new_centroids):
        distance = [euclidean_distance(old_centroids[i], new_centroids[i]) for i in range(self.k)]
        return sum(distance) == 0

    def get_label(self, clusters):
        labels = np.empty(self.n_samples)
        for idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                # print(sample_idx)
                labels[sample_idx] = idx

        return labels

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, idx in enumerate(self.clusters):
            point = self.x[idx].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)
        plt.show()