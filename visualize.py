import matplotlib.pyplot as plt


def plot_clusters(
    X,
    labels,
    centroids=None,
    title="Clusters"
):

    plt.figure(figsize=(8, 6))

    plt.scatter(

        X[:, 0],
        X[:, 1],

        c=labels,

        cmap="viridis",

        s=20

    )

    if centroids is not None:

        plt.scatter(

            centroids[:, 0],
            centroids[:, 1],

            c="red",

            marker="x",

            s=200,

            linewidths=3

        )

    plt.title(title)

    plt.show()