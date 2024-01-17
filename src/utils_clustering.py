import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
from datetime import datetime

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


def evaluate_clustering(X, labels_true, labels_pred, clustering_name, dataset_name, results_path):
    """
    Evaluates the clustering performance using various metrics and saves the results to a CSV file.

    :param X: Feature set.
    :param labels_true: Ground truth labels.
    :param labels_pred: Predicted cluster labels.
    :param clustering_name: Name of the clustering algorithm.
    :param dataset_name: Name of the dataset.
    :param results_path: Path to save the results CSV file.
    """
    # Collecting metrics
    results = {
        'Timestamp': datetime.now(),
        'Dataset': dataset_name,
        'Clustering Algorithm': clustering_name,
        'AMI': metrics.adjusted_mutual_info_score(labels_true, labels_pred),
        'ARI': metrics.adjusted_rand_score(labels_true, labels_pred),
        'Calinski-Harabasz Score': metrics.calinski_harabasz_score(X, labels_pred),
        'Davies-Bouldin Score': metrics.davies_bouldin_score(X, labels_pred),
        'Completeness Score': metrics.completeness_score(labels_true, labels_pred),
        'Fowlkes-Mallows Score': metrics.fowlkes_mallows_score(labels_true, labels_pred),
        'Homogeneity': metrics.homogeneity_score(labels_true, labels_pred),
        'Completeness': metrics.completeness_score(labels_true, labels_pred),
        'V-Measure': metrics.v_measure_score(labels_true, labels_pred),
        'Mutual Information': metrics.mutual_info_score(labels_true, labels_pred),
        'Normalized Mutual Information': metrics.normalized_mutual_info_score(labels_true, labels_pred),
        'Rand Score': metrics.rand_score(labels_true, labels_pred),
        'Silhouette Score': metrics.silhouette_score(X, labels_pred),
    }

    # Print results
    for key, value in results.items():
        print(f"{key}: {value}")

    # Save to CSV
    df = pd.DataFrame([results])
    df.to_csv(results_path, mode='a', header=not pd.io.common.file_exists(results_path), index=False)


def load_labels_from_file(file_path, labels_pred_len):
    """
    Loads clustering labels from a text file, ignoring the header and metadata.

    :param file_path: Path to the file containing the labels.
    :return: List of labels as integers.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Skipping the header and metadata, start reading from the line after '-----'
    start_index = lines.index('-------------------------------------\n') + 1
    labels_true = [int(line.strip()) for line in lines[start_index:]]
    if labels_pred_len != len(labels_true):
        raise ValueError(f"This is a custom error raised by the developer./ Please check the file {file_path} or labels "
                         f"definition.")

    return labels_true


# Function to visualize the biclusters
def visualize_biclusters(data, row_labels, col_labels, title='Biclustering Visualization'):
    sorted_data = data[np.argsort(row_labels), :]
    sorted_data = sorted_data[:, np.argsort(col_labels)]

    plt.figure(figsize=(12, 8))  # Increase figure size
    plt.imshow(sorted_data, aspect='auto', cmap='viridis')
    plt.colorbar()  # Add a color bar
    plt.grid(False)  # Disable grid (set to True to enable)
    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()


def evaluate_biclustering(data,  labels_true, bicluster, clustering_name, dataset_name,results_path):
    """
    Evaluates the biclustering performance using various metrics.

    :param data: Feature set.
    :param labels_true: Ground truth labels for traditional clustering metrics.
    :param bicluster: Fitted biclustering object.
    :param clustering_name: Name of the clustering algorithm.
    :param dataset_name: Name of the dataset.
    :param results_path: Path to save the results CSV file.
    """
    # Collecting metrics
    results = {
        'Timestamp': datetime.now(),
        'Dataset': dataset_name,
        'Clustering Algorithm': clustering_name,
        'AMI': metrics.adjusted_mutual_info_score(labels_true, bicluster.row_labels_),
        'Silhouette Score': metrics.silhouette_score(data, bicluster.row_labels_),
        # 'Consensus Score': metrics.consensus_score(bicluster.biclusters_, bicluster.biclusters_),
    }

    # Print results
    for key, value in results.items():
        print(f"{key}: {value}")

    # Save to CSV
    df = pd.DataFrame([results])
    df.to_csv(results_path, mode='a', header=not pd.io.common.file_exists(results_path), index=False)