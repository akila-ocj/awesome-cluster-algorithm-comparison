import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def plot_clusters(data, labels):
    """
    Function to plot the results of Agglomerative Clustering.

    :param data: pandas DataFrame with the processed data.
    :param labels: array-like, cluster labels for each point.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title('Agglomerative Clustering Visualization')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def evaluate_clustering(data, labels):
    """
    Evaluates the clustering performance using silhouette score.

    :param data: DataFrame with preprocessed data.
    :param labels: Array of cluster labels.
    :return: Silhouette score.
    """
    score = silhouette_score(data, labels)
    print(f"Silhouette Score: {score}")
    return score
