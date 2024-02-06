import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from scipy.optimize import linear_sum_assignment


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


def evaluate_clustering(X, labels_true, labels_pred, clus_algo_name, dataset_name, results_path):
    """
    Evaluates the clustering performance using various metrics and saves the results to a CSV file.

    :param X: Feature set.
    :param labels_true: Ground truth labels.
    :param labels_pred: Predicted cluster labels.
    :param clus_algo_name: Name of the clustering algorithm.
    :param dataset_name: Name of the dataset.
    :param results_path: Path to save the results CSV file.
    """
    results = {
        'Timestamp': datetime.now(),
        'Dataset': dataset_name,
        'Clustering Algorithm': clus_algo_name,
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
    df.to_csv(results_path, mode='a', header=not pd.io.common.file_exists(results_path), index=False)


def map_clusters_to_ground_truth(labels_true, labels_pred):
    """
    Maps clustering algorithm output to ground truth labels using the Hungarian algorithm.

    :param labels_true: Ground truth labels.
    :param labels_pred: Predicted cluster labels.
    :return: Remapped predicted labels.
    """
    # Calculate the confusion matrix
    cm = confusion_matrix(labels_true, labels_pred)
    # Apply the Hungarian algorithm to the negative confusion matrix for maximum matching
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Create a new array to hold the remapped predicted labels
    remapped_labels_pred = np.zeros_like(labels_pred)
    # For each original cluster index, find the new label (according to the Hungarian algorithm)
    # and assign it in the remapped labels array
    for original_cluster, new_label in zip(col_ind, row_ind):
        remapped_labels_pred[labels_pred == original_cluster] = new_label

    return remapped_labels_pred


def map_clusters_to_ground_truth_dbscan(labels_true, labels_pred):
    """
    Adjusted mapping for DBSCAN output to ground truth labels, excluding noise points.
    """
    # Ensure labels_true and labels_pred are numpy arrays for advanced indexing
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)

    # Filter out noise points (-1 labels) from labels_pred for mapping
    valid_idx = labels_pred != -1
    labels_pred_filtered = labels_pred[valid_idx]
    labels_true_filtered = labels_true[valid_idx]

    # Calculate the confusion matrix without noise points
    cm = confusion_matrix(labels_true_filtered, labels_pred_filtered)
    # Apply the Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Initialize remapped labels array with -1 for noise points
    remapped_labels_pred = -1 * np.ones_like(labels_pred)
    # Map valid clusters excluding noise
    for original, new in zip(col_ind, row_ind):
        # Find indices in the filtered prediction that match the original cluster
        indices = np.where(labels_pred_filtered == original)[0]
        # For each of these indices, update the corresponding entry in the full remapped prediction array
        for idx in indices:
            # Convert filtered index back to original index
            original_idx = np.where(valid_idx)[0][idx]
            remapped_labels_pred[original_idx] = new

    return remapped_labels_pred

def generate_confusion_matrix(labels_true, labels_pred, n_classes):
    """
    Generates and prints the confusion matrix and accuracies for each cluster.

    :param labels_true: Ground truth labels.
    :param labels_pred: Predicted cluster labels.
    :param n_classes: Number of classes or clusters.
    """

    # Adjust the labels parameter to include the new range of labels
    labels_range = list(range(1, n_classes + 1))  # Adjusted for 1-15

    # Compute confusion matrix
    cm = confusion_matrix(labels_true, labels_pred, labels=labels_range)
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
        raise ValueError(
            f"This is a custom error raised by the developer./ Please check the file {file_path} or labels "
            f"definition.")

    return labels_true
