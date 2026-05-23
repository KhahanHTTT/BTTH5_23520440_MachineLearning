from dataset import *

from kmeans import KMeans

from gmm import GMM

from visualize import plot_clusters

import cv2

import matplotlib.pyplot as plt


# ASSIGNMENT 1 - KMEANS

print("=" * 50)

print("ASSIGNMENT 1 - KMEANS")

print("=" * 50)

X = create_assignment1_dataset()

model = KMeans(
    n_clusters=3,
    max_iters=100
)

labels = model.fit(X)

plot_clusters(
    X,
    labels,
    model.centroids,
    title="Assignment 1 - KMeans"
)

print(
    "Random centroid initialization affects convergence."
)


# ASSIGNMENT 2 - KMEANS

print("=" * 50)

print("ASSIGNMENT 2 - KMEANS")

print("=" * 50)

X = create_assignment2_dataset()

model = KMeans(
    n_clusters=3,
    max_iters=100
)

labels = model.fit(X)

plot_clusters(
    X,
    labels,
    model.centroids,
    title="Assignment 2 - KMeans"
)

print(
    "Different cluster sizes reduce clustering quality."
)


# ASSIGNMENT 3 - KMEANS

print("=" * 50)

print("ASSIGNMENT 3 - KMEANS")

print("=" * 50)

X = create_assignment3_dataset()

model = KMeans(
    n_clusters=3,
    max_iters=100
)

labels = model.fit(X)

plot_clusters(
    X,
    labels,
    model.centroids,
    title="Assignment 3 - KMeans"
)

print(
    "Elongated Gaussian distribution affects KMeans."
)


# ASSIGNMENT 1 - GMM

print("=" * 50)

print("ASSIGNMENT 1 - GMM")

print("=" * 50)

X = create_assignment1_dataset()

model = GMM(
    n_components=3,
    max_iters=100
)

labels = model.fit(X)

plot_clusters(
    X,
    labels,
    model.means,
    title="Assignment 1 - GMM"
)


# =========================================================
# ASSIGNMENT 2 - GMM IMAGE SEGMENTATION
# =========================================================

print("=" * 50)

print("ASSIGNMENT 2 - GMM IMAGE SEGMENTATION")

print("=" * 50)

image = cv2.imread("cow.jpg")

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# pixels = image.reshape((-1, 3))
pixels = image.reshape((-1, 3)).astype(np.float64)

pixels = pixels / 255.0

model = GMM(
    n_components=3,
    max_iters=20
)

labels = model.fit(pixels)

segmented = model.means[labels]

segmented = segmented.reshape(image.shape)

# segmented = segmented.astype("uint8")
segmented = (segmented * 255).astype("uint8")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)

plt.imshow(image)

plt.title("Original Image")

plt.subplot(1, 2, 2)

plt.imshow(segmented)

plt.title("Segmented Image")

plt.show()