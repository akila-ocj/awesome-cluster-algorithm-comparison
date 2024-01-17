import matplotlib.pyplot as plt
from sklearn import metrics

def plot_clusters(data, labels_pred, title='Clustering Visualization'):
    """
    Function to plot the results of a clustering algorithm.

    :param data: pandas DataFrame with the processed data.
    :param labels: array-like, cluster labels for each point.
    :param title: string, title for the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels_pred, cmap='viridis', marker='o')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def evaluate_clustering(X, labels_true, labels_pred, title='Clustering Evaluation'):
    """
    Evaluates the clustering performance using various metrics.

    :param X: Feature set.
    :param labels_true: Ground truth labels.
    :param labels_pred: Predicted cluster labels.
    :param title: Title for the evaluation printout.
    """
    print(f"{title}\n" + "=" * len(title))

    # Adjusted Mutual Information
    ami = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    print(f"Adjusted Mutual Information (AMI): {ami}")

    # Adjusted Rand Index
    ari = metrics.adjusted_rand_score(labels_true, labels_pred)
    print(f"Adjusted Rand Index (ARI): {ari}")

    # Calinski and Harabasz Score
    chs = metrics.calinski_harabasz_score(X, labels_pred)
    print(f"Calinski and Harabasz Score: {chs}")

    # Davies-Bouldin Score
    dbs = metrics.davies_bouldin_score(X, labels_pred)
    print(f"Davies-Bouldin Score: {dbs}")

    # Completeness Score
    comp_score = metrics.completeness_score(labels_true, labels_pred)
    print(f"Completeness Score: {comp_score}")

    # Fowlkes-Mallows Score
    fms = metrics.fowlkes_mallows_score(labels_true, labels_pred)
    print(f"Fowlkes-Mallows Score: {fms}")

    # Homogeneity, Completeness, V-Measure
    homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
    print(f"Homogeneity: {homogeneity}, Completeness: {completeness}, V-Measure: {v_measure}")

    # Homogeneity Score
    homogeneity_score = metrics.homogeneity_score(labels_true, labels_pred)
    print(f"Homogeneity Score: {homogeneity_score}")

    # Mutual Information
    mi = metrics.mutual_info_score(labels_true, labels_pred)
    print(f"Mutual Information: {mi}")

    # Normalized Mutual Information
    nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    print(f"Normalized Mutual Information: {nmi}")

    # Rand Score
    rand_score = metrics.rand_score(labels_true, labels_pred)
    print(f"Rand Score: {rand_score}")

    # Silhouette Score
    silhouette = metrics.silhouette_score(X, labels_pred)
    print(f"Silhouette Score: {silhouette}")

    # V-Measure Score
    v_measure_score = metrics.v_measure_score(labels_true, labels_pred)
    print(f"V-Measure Score: {v_measure_score}")

    print("\n" + "-" * 40 + "\n")


def load_labels_from_file(file_path):
    """
    Loads clustering labels from a text file, ignoring the header and metadata.

    :param file_path: Path to the file containing the labels.
    :return: List of labels as integers.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Skipping the header and metadata, start reading from the line after '-----'
    start_index = lines.index('-------------------------------------\n') + 1
    labels = [int(line.strip()) for line in lines[start_index:]]

    return labels