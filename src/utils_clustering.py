import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

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
    results = {
        'Timestamp': datetime.now(),
        'Dataset': dataset_name,
        'Clustering Algorithm': clustering_name,
        'Accuracy': accuracy_score(labels_true, labels_pred)
    }

    # Print results
    for key, value in results.items():
        if key == 'Confusion Matrix':
            print(f"{key}:\n{value}")
        else:
            print(f"{key}: {value}")

    # Save to CSV
    df = pd.DataFrame([results])
    # df.to_csv(results_path, mode='a', header=not pd.io.common.file_exists(results_path), index=False)

def generate_confusion_matrix(labels_true, labels_pred, n_classes):
    """
    Generates and prints the confusion matrix and accuracies for each cluster.

    :param labels_true: Ground truth labels.
    :param labels_pred: Predicted cluster labels.
    :param n_classes: Number of classes or clusters.
    """
    # Compute confusion matrix
    cm = confusion_matrix(labels_true, labels_pred, labels=np.arange(n_classes))
    np.set_printoptions(precision=2)

    # Print confusion matrix
    print("Confusion Matrix:")
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Compute accuracies for each cluster, safely handling division by zero
    row_sums = np.sum(cm, axis=1)
    safe_divisor = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
    cluster_accuracies = np.diagonal(cm) / safe_divisor

    # Handling potential division by zero by replacing inf and NaN values with 0
    cluster_accuracies = np.nan_to_num(cluster_accuracies)  # Replace NaN and inf with 0
    print("Each cluster's accuracy indicates how well the clustering algorithm has grouped the data points,")
    print("compared to the ground truth labels. Higher accuracy means a closer match to the expected grouping.\n")

    # Print each cluster's accuracy
    for i, accuracy in enumerate(cluster_accuracies, start=1):
        print(f"Cluster {i} Accuracy: {accuracy * 100:.2f}%")


    # Compute overall accuracy
    overall_accuracy = accuracy_score(labels_true, labels_pred)
    # Print overall accuracy
    print(f"\nOverall Accuracy: {overall_accuracy * 100:.2f}%")
    print("This represents the proportion of all data points that were correctly grouped by the clustering algorithm.")


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
