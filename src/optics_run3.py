import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

import os
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, Birch, OPTICS, MeanShift, AgglomerativeClustering
import time
from sklearn.model_selection import ParameterSampler
from typing import Union, Tuple
from os import PathLike
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
import ast

n_iter = 1000
MAJOR_MINOR_VERSION = '1.5'
from sklearn.model_selection import KFold

n_splits = 3  # Number of folds
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


def evaluate_clustering(X, labels_true, labels_pred, clus_algo_name, dataset_name, results_path, algorithm_details,
                        training_time, prediction_time):
    """
    Evaluates the clustering performance using various metrics and saves the results to a CSV file.

    :param X: Feature set.
    :param labels_true: Ground truth labels.
    :param labels_pred: Predicted cluster labels.
    :param clus_algo_name: Name of the clustering algorithm.
    :param dataset_name: Name of the dataset.
    :param results_path: Path to save the results CSV file.
    """
    # Ensure there are at least 2 unique labels before calculating certain metrics
    unique_labels = np.unique(labels_pred)

    # Initialize default values for scores that require multiple clusters
    calinski_harabasz_score = np.nan
    davies_bouldin_score = np.nan
    silhouette_score = np.nan

    if len(unique_labels) > 1:
        calinski_harabasz_score = metrics.calinski_harabasz_score(X, labels_pred)
        davies_bouldin_score = metrics.davies_bouldin_score(X, labels_pred)
        silhouette_score = metrics.silhouette_score(X, labels_pred)

    results = {
        'Timestamp': datetime.now(),
        'Dataset': dataset_name,
        'Clustering Algorithm': clus_algo_name,
        'Algorithm Details': algorithm_details,
        'Training Time': training_time,
        'Prediction Time': prediction_time,
        'AMI': metrics.adjusted_mutual_info_score(labels_true, labels_pred),
        'ARI': metrics.adjusted_rand_score(labels_true, labels_pred),
        'Calinski-Harabasz Score': calinski_harabasz_score,
        'Davies-Bouldin Score': davies_bouldin_score,
        'Completeness Score': metrics.completeness_score(labels_true, labels_pred),
        'Fowlkes-Mallows Score': metrics.fowlkes_mallows_score(labels_true, labels_pred),
        'Homogeneity': metrics.homogeneity_score(labels_true, labels_pred),
        'Completeness': metrics.completeness_score(labels_true, labels_pred),
        'V-Measure': metrics.v_measure_score(labels_true, labels_pred),
        'Mutual Information': metrics.mutual_info_score(labels_true, labels_pred),
        'Normalized Mutual Information': metrics.normalized_mutual_info_score(labels_true, labels_pred),
        'Silhouette Score': silhouette_score,
        'Accuracy': accuracy_score(labels_true, labels_pred)

    }

    # # Print results
    # for key, value in results.items():
    #     if key == 'Confusion Matrix':
    #         print(f"{key}:\n{value}")
    #     else:
    #         print(f"{key}: {value}")

    # Save to CSV
    df = pd.DataFrame([results])
    df.to_csv(results_path, mode='a', header=not os.path.exists(results_path), index=False)


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
    # if labels_pred_len != len(labels_true):
    #     raise ValueError(
    #         f"This is a custom error raised by the developer./ Please check the file {file_path} or labels "
    #         f"definition.")

    return labels_true


def preprocess_data(df):
    """
    Function to preprocess data by normalizing it.

    :param df: pandas DataFrame with raw data.
    :return: pandas DataFrame with processed (normalized) data.
    """
    processed_df = df.copy()

    # Normalize the data
    # For each column, subtract the minimum and divide by the range.
    for column in processed_df.columns:
        min_value = processed_df[column].min()
        max_value = processed_df[column].max()
        processed_df[column] = (processed_df[column] - min_value) / (max_value - min_value)

    return processed_df


def find_next_version_number(base_path: str, algorithm_name: str, major_minor_version: str) -> str:
    """
    Finds the next version number based on existing directories within the algorithm name directory.
    Args:
    - base_path: The base directory path where algorithms are stored.
    - algorithm_name: The name of the algorithm.
    - major_minor_version: The major and minor version components (e.g., '1.0').

    Returns:
    - The next version as a string (e.g., '1.0.3').
    """
    algorithm_path = os.path.join(base_path, algorithm_name)
    if not os.path.exists(algorithm_path):
        return major_minor_version + '.1'  # Start with version .1 if no directory exists

    # List all version directories and filter by the major_minor_version prefix
    version_dirs = [d for d in os.listdir(algorithm_path) if os.path.isdir(os.path.join(algorithm_path, d))]
    version_nums = [d.replace(major_minor_version + '.', '') for d in version_dirs if d.startswith(major_minor_version)]

    if not version_nums:
        return major_minor_version + '.1'  # Start with version .1 if no matching directories

    # Find the highest current version number
    latest_version = max([int(num) for num in version_nums if num.isdigit()], default=0)

    # Return the next version number
    return major_minor_version + '.' + str(latest_version + 1)


# Function to create directories and return the path for results
def create_dirs_and_get_results_path(base_path: Union[str, PathLike],
                                     algorithm_name: str,
                                     version: str,
                                     dataset_dir: str,
                                     dataset_name: str,
                                     filename: str) -> str:
    directory_path = os.path.join(base_path, algorithm_name, version, dataset_dir, dataset_name)
    os.makedirs(directory_path, exist_ok=True)
    return os.path.join(directory_path, filename)


def calculate_grid_size(space):
    # For each parameter, count the number of unique values and multiply them
    return np.prod([len(values) for values in space.values()])


# Functions
def train_and_time(clustering_model, train_data):
    """Train the model and measure training time."""
    start_time = time.time()
    try:
        clustering_model.fit(train_data)
    except Exception as e:
        return None, None
    end_time = time.time()
    return clustering_model, end_time - start_time


def predict_and_time(clustering_model, data):
    """Predict using the model and measure prediction time."""
    start_time = time.time()
    if ALGORITHM_NAME == 'DBSCAN' or ALGORITHM_NAME == 'OPTICS' or ALGORITHM_NAME == 'AgglomerativeClustering':
        labels_pred = clustering_model.fit_predict(data)
    else:
        labels_pred = clustering_model.predict(data)
    end_time = time.time()
    return labels_pred, end_time - start_time


def evaluate_and_log(clustering_model, X_train, X_validate, labels_true, results_path):
    """Evaluate the clustering and log the results."""
    algorithm_details = str(clustering_model.get_params())
    _, training_time = train_and_time(clustering_model, X_train)
    labels_pred, prediction_time = predict_and_time(clustering_model, X_validate)
    labels_pred = map_clusters_to_ground_truth(labels_true, labels_pred)
    evaluate_clustering(X=X_validate, labels_true=labels_true, labels_pred=labels_pred,
                        clus_algo_name=ALGORITHM_NAME, dataset_name=DATASET_FILE_NAME,
                        results_path=results_path, algorithm_details=algorithm_details,
                        training_time=training_time, prediction_time=prediction_time)


def generate_expanded_search_domains(data_size, feature_count):
    """
    Generates expanded search domains for the OPTICS clustering algorithm hyperparameters.

    Parameters:
    - data_size: int, the total number of points in the dataset.
    - feature_count: int, the number of features in the dataset.

    Returns:
    - A dictionary with expanded search domains for 'min_samples', 'xi', and 'min_cluster_size'.
    """

    # Expand dynamic range for min_samples based on data size and feature count
    min_samples_start = 2
    # Increase the upper limit for min_samples significantly
    min_samples_end = max(100, int(data_size * 0.1), feature_count * 3)  # More aggressive adjustment
    min_samples_domain = np.linspace(min_samples_start, min_samples_end, num=20,
                                     dtype=int).tolist()  # Increase number of points

    # Expand xi search domain to cover a wider range of values
    xi_domain = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # Adding more granularity and higher values

    # Expand min_cluster_size search domain, considering both fractions and absolute numbers
    # Allow for even smaller fractions and higher absolute numbers
    min_cluster_size_fraction_domain = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25]
    min_cluster_size_absolute_start = int(data_size * 0.005)  # Allowing for smaller clusters
    min_cluster_size_absolute_end = max(100, int(data_size * 0.2))  # Allowing for larger clusters
    min_cluster_size_absolute_domain = np.linspace(min_cluster_size_absolute_start, min_cluster_size_absolute_end,
                                                   num=20, dtype=int).tolist()  # Increase number of points

    # Combine fraction and absolute domains for min_cluster_size
    min_cluster_size_domain = sorted(
        set(min_cluster_size_fraction_domain + min_cluster_size_absolute_domain))  # Remove duplicates and sort

    return {
        'min_samples': min_samples_domain,
        'xi': xi_domain,
        'min_cluster_size': min_cluster_size_domain
    }


# Exponent value generation technique
# Generate 20 random exponent values uniformly between -4 and 1 for DBSCAN's eps
eps_exponents = np.random.uniform(-5, 2, 20000)
eps_values = 10 ** eps_exponents

warnings.filterwarnings("ignore")
current_directory = os.path.join('/opt', 'home', 's3934056')
DATASET_DIRS = [
    'A-sets',
    # 'Birch-sets',
    'DIM-sets-high',
    # 'G2-sets',
    'S-sets',
    'Unbalance'
]
# DATASET_DIRS = ['A-sets']


metrics_file_path = os.path.join(current_directory, 'results', 'metrics')
# Determine the next version number automatically


algorithms = [
    'OPTICS'
]

for algorithm_name in algorithms:
    ALGORITHM_NAME = algorithm_name
    VERSION = find_next_version_number(metrics_file_path, ALGORITHM_NAME, MAJOR_MINOR_VERSION)
    for DATASET_DIR in DATASET_DIRS:
        # Specify the raw and processed data directory paths
        raw_directory_path = os.path.join(current_directory, 'data', 'raw', DATASET_DIR)

        processed_directory_path = os.path.join(current_directory, 'data', 'processed', DATASET_DIR)

        os.makedirs(processed_directory_path, exist_ok=True)  # This creates the directory if it does not exist

        # Get a list of all files in the directory (excluding directories)
        files = [f for f in os.listdir(raw_directory_path) if os.path.isfile(os.path.join(raw_directory_path, f))]
        total_files = len(files)

        # Run preprocessing step for all files in the specified directory
        for index, filename in enumerate(files, start=1):
            FILE_NAME = filename.split('.')[0]
            raw_file_path = os.path.join(raw_directory_path, f'{FILE_NAME}.txt')
            processed_file_path = os.path.join(processed_directory_path, f'{FILE_NAME}.txt')

            # Check if the processed file already exists
            if not os.path.isfile(processed_file_path):
                # The regular expression '\s+' can be used to match one or more spaces
                data = pd.read_csv(raw_file_path, sep="\s+", header=None, names=['X', 'Y'])
                # Remove rows with missing values:
                processed_data = preprocess_data(data)

                # Save the processed data to a CSV file
                processed_data.to_csv(processed_file_path, index=False)

        # Run clustering algorithm for all files in the specified directory
        for index, filename in enumerate(files, start=1):
            if os.path.isfile(os.path.join(raw_directory_path, filename)):
                # print(f"[{index}/{total_files}] {filename.split('.')[0]}")

                DATASET_FILE_NAME = filename.split('.')[0]
                LABELS_FILE_NAME = f'{DATASET_FILE_NAME}-gt.pa'

                # Read labels
                labels_true = load_labels_from_file(
                    os.path.join(current_directory, 'data', 'label', DATASET_DIR, LABELS_FILE_NAME), 15)

                # Get the number of clusters form the ground truth
                N_CLUSTERS = len(set(labels_true))
                raw_file_path = os.path.join(current_directory, 'data', 'raw', DATASET_DIR, f'{DATASET_FILE_NAME}.txt')

                raw_data = pd.read_csv(raw_file_path, sep='\s+', )
                N_DIMENSIONS = raw_data.shape[1]

                # hyperparameter_domains = {
                #     'OPTICS': {
                #         'min_samples': [10, 20, 50],
                #         'max_eps': [1, 5, 7.5, np.inf],
                #         'metric': ['euclidean', 'cosine'],
                #         'xi': np.linspace(0.01, 0.1, 5),
                #         'min_cluster_size': [None, 5, 10, 20]
                #     }
                # }

                # Read processed data
                # processed_file_path = rf'data\processed\{DATASET_DIR}\{DATASET_FILE_NAME}.txt'
                processed_file_path = os.path.join(current_directory, 'data', 'processed', DATASET_DIR,
                                                   f'{DATASET_FILE_NAME}.txt')
                processed_data = pd.read_csv(processed_file_path)
                hyperparameter_domain = generate_expanded_search_domains(len(processed_data), N_DIMENSIONS)
                grid_size = calculate_grid_size(hyperparameter_domain)

                # Adjusted for creating directories and files accordingly
                # For Hyperparameter Tuning Logs
                tuning_results_filename = 'hyperparameter_tuning_logs.csv'
                tuning_results_path = create_dirs_and_get_results_path(metrics_file_path, ALGORITHM_NAME, VERSION,
                                                                       DATASET_DIR, DATASET_FILE_NAME,
                                                                       tuning_results_filename)

                # For Final Results
                final_results_filename = 'results.csv'
                final_results_path = create_dirs_and_get_results_path(metrics_file_path, ALGORITHM_NAME, VERSION,
                                                                      DATASET_DIR, DATASET_FILE_NAME,
                                                                      final_results_filename)

                parameter_sampler = ParameterSampler(hyperparameter_domain, n_iter=n_iter, random_state=42)

                labels_true = np.array(labels_true)

                for fold, (train_index, test_index) in enumerate(kf.split(processed_data, labels_true)):
                    X_train, X_test = processed_data.iloc[train_index], processed_data.iloc[test_index]
                    y_train, y_test = labels_true[train_index], labels_true[test_index]

                    for params in parameter_sampler:
                        if algorithm_name == 'OPTICS':
                            model = OPTICS(**params)
                        else:
                            raise ValueError("Unsupported algorithm")

                        evaluate_and_log(model, X_train, X_test, y_test, tuning_results_path)

                # Read hyperparameter tuning logs
                csv_content = pd.read_csv(tuning_results_path)

                # Find the record with the highest accuracy
                max_accuracy_record = csv_content.loc[csv_content['Accuracy'].idxmax()]

                df = pd.DataFrame([max_accuracy_record])
                df.to_csv(final_results_path, mode='a', header=not os.path.exists(final_results_path), index=False)
