import numpy as np

# Generate Gaussian Data

def generate_gaussian_data(mean, covariance, n_samples):

    data = np.random.multivariate_normal(
        mean,
        covariance,
        n_samples
    )

    return data


# Assignment 1 Dataset

def create_assignment1_dataset():

    np.random.seed(42)

    sigma = np.array([
        [1, 0],
        [0, 1]
    ])

    x1 = generate_gaussian_data([2, 2], sigma, 200)

    x2 = generate_gaussian_data([8, 3], sigma, 200)

    x3 = generate_gaussian_data([3, 6], sigma, 200)

    X = np.vstack((x1, x2, x3))

    return X


# Assignment 2 Dataset

def create_assignment2_dataset():

    np.random.seed(42)

    sigma = np.array([
        [1, 0],
        [0, 1]
    ])

    x1 = generate_gaussian_data([2, 2], sigma, 200)

    x2 = generate_gaussian_data([8, 3], sigma, 200)

    x3 = generate_gaussian_data([3, 6], sigma, 1000)

    X = np.vstack((x1, x2, x3))

    return X


# Assignment 3 Dataset

def create_assignment3_dataset():

    np.random.seed(42)

    sigma1 = np.array([
        [1, 0],
        [0, 1]
    ])

    sigma2 = np.array([
        [10, 0],
        [0, 1]
    ])

    x1 = generate_gaussian_data([2, 2], sigma1, 200)

    x2 = generate_gaussian_data([8, 3], sigma1, 200)

    x3 = generate_gaussian_data([3, 6], sigma2, 200)

    X = np.vstack((x1, x2, x3))

    return X