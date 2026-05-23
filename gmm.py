import numpy as np


class GMM:

    def __init__(
        self,
        n_components=3,
        max_iters=100,
        tol=1e-4
    ):

        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol


    # Initialize Parameters

    def initialize_parameters(self, X):

        n_samples, n_features = X.shape

        self.weights = np.ones(
            self.n_components
        ) / self.n_components

        random_idx = np.random.choice(
            n_samples,
            self.n_components,
            replace=False
        )

        self.means = X[random_idx]

        self.covariances = np.array([

            np.eye(n_features)

            for _ in range(self.n_components)

        ])


    # Gaussian Distribution

    def gaussian_distribution(
        self,
        X,
        mean,
        covariance
    ):

        n_features = X.shape[1]

        covariance += 1e-6 * np.eye(n_features)

        determinant = np.linalg.det(covariance)

        inverse = np.linalg.inv(covariance)

        coefficient = 1 / np.sqrt(

            ((2 * np.pi) ** n_features)
            * determinant

        )

        diff = X - mean

        exponent = -0.5 * np.sum(

            (diff @ inverse) * diff,
            axis=1

        )

        probability = coefficient * np.exp(exponent)

        return probability


    # E-Step

    def expectation_step(self, X):

        n_samples = X.shape[0]

        responsibilities = np.zeros(

            (n_samples, self.n_components)

        )

        for k in range(self.n_components):

            responsibilities[:, k] = (

                self.weights[k]

                * self.gaussian_distribution(

                    X,
                    self.means[k],
                    self.covariances[k]

                )
            )

        responsibilities /= responsibilities.sum(
            axis=1,
            keepdims=True
        )

        return responsibilities


    # M-Step

    def maximization_step(
        self,
        X,
        responsibilities
    ):

        n_samples = X.shape[0]

        Nk = np.sum(
            responsibilities,
            axis=0
        )

        self.weights = Nk / n_samples

        self.means = (

            responsibilities.T @ X

        ) / Nk[:, np.newaxis]

        covariances = []

        for k in range(self.n_components):

            diff = X - self.means[k]

            covariance = (

                responsibilities[:, k][:, np.newaxis]
                * diff

            ).T @ diff

            covariance /= Nk[k]

            covariances.append(covariance)

        self.covariances = np.array(
            covariances
        )


    # Fit

    def fit(self, X):

        self.initialize_parameters(X)

        previous_log_likelihood = 0

        for iteration in range(self.max_iters):

            # E-Step

            responsibilities = self.expectation_step(X)

            # M-Step

            self.maximization_step(
                X,
                responsibilities
            )

            # Log Likelihood

            likelihood = np.zeros(X.shape[0])

            for k in range(self.n_components):

                likelihood += (

                    self.weights[k]

                    * self.gaussian_distribution(

                        X,
                        self.means[k],
                        self.covariances[k]

                    )
                )

            log_likelihood = np.sum(
                np.log(likelihood + 1e-10)
            )

            if abs(
                log_likelihood
                - previous_log_likelihood
            ) < self.tol:

                print(
                    f"Converged at iteration {iteration}"
                )

                break

            previous_log_likelihood = log_likelihood

        labels = np.argmax(
            responsibilities,
            axis=1
        )

        return labels