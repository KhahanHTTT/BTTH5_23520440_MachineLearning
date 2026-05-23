import numpy as np


class KMeans:

    def __init__(
        self,
        n_clusters=3,
        max_iters=100,
        tol=1e-4
    ):

        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol

        self.centroids = None
        self.labels = None

    # Initialize Centroids

    def initialize_centroids(self, X):

        n_samples = X.shape[0]

        random_idx = np.random.choice(
            n_samples,
            self.n_clusters,
            replace=False
        )

        self.centroids = X[random_idx]

    # Compute Distance

    def compute_distance(self, X):

        distances = np.linalg.norm(
            X[:, np.newaxis] - self.centroids,
            axis=2
        )

        return distances

    # E-Step

    def expectation_step(self, X):

        distances = self.compute_distance(X)

        self.labels = np.argmin(
            distances,
            axis=1
        )

    # M-Step

    def maximization_step(self, X):

        new_centroids = []

        for k in range(self.n_clusters):

            cluster_points = X[self.labels == k]

            if len(cluster_points) == 0:

                centroid = self.centroids[k]

            else:

                centroid = np.mean(
                    cluster_points,
                    axis=0
                )

            new_centroids.append(centroid)

        new_centroids = np.array(new_centroids)

        return new_centroids

    # Fit

    def fit(self, X):

        self.initialize_centroids(X)

        for iteration in range(self.max_iters):

            # E-Step
            self.expectation_step(X)

            # M-Step
            new_centroids = self.maximization_step(X)

            # Check convergence
            shift = np.linalg.norm(
                new_centroids - self.centroids
            )

            self.centroids = new_centroids

            if shift < self.tol:

                print(
                    f"Converged at iteration {iteration}"
                )

                break

        return self.labels