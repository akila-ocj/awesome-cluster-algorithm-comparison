# Clustering Analysis Project

## Overview
This repository contains the implementation and analysis of various clustering algorithms. It's structured to facilitate exploratory data analysis, algorithm implementation, and evaluation in a clear, modular, and reproducible manner.

Here are the identified clustering algorithms along with their corresponding implementations. The column 'Algorithm and Original Paper Name' includes the title of the original paper and the total number of citations it has received. For instance, the K-means algorithm was introduced in a paper titled 'Some methods for classification and analysis of multivariate observations,' which has accumulated 37,287 citations to date.
<br>
<br>
| Rank | Algorithm | Original Paper Name | Cited By | isClusteringAlgorithm | Implementation |
|------|-----------|----------------------|----------|-----------------------|----------------|
| 1    | k-means   | Some methods for classification and analysis of multivariate observations | 37287    | TRUE                  | sklearn.cluster.KMeans |
| 2    | DBSCAN    | A density-based algorithm for discovering clusters in large spatial databases with noise | 30900    | TRUE                  | sklearn.cluster.DBSCAN |
| 3    | BIRCH     | an efficient data clustering method for very large databases | 7475     | TRUE                  | sklearn.cluster.Birch |
| 4    | Fuzzy C Means | FCM: The fuzzy c-means clustering algorithm | 7462     | TRUE                  | pyclustering.cluster.fcm.fcm |
| 5    | OPTICS    | OPTICS: Ordering points to identify the clustering structure | 6134     | TRUE                  | sklearn.cluster.OPTICS |
| 6    | Mean Shift | Mean shift, mode seeking, and clustering | 5742     | TRUE                  | sklearn.cluster.MeanShift |
| 7    | CURE      | CURE: An efficient clustering algorithm for large databases | 4175     | TRUE                  | pyclustering.cluster.cure.cure |
| 8    | bisecting K-Means | A comparison of document clustering techniques | 4094     | TRUE                  | sklearn.cluster.BisectingKMeans |
| 9    | CLIQUE    | Automatic subspace clustering of high dimensional data for data mining applications | 3933     | TRUE                  | pyclustering.cluster.clique.clique |
| 10   | Ward's hierarchical agglomerative clustering | Ward's hierarchical agglomerative clustering method: which algorithms implement Ward's criterion? | 3168     | TRUE                  | sklearn.cluster.AgglomerativeClustering |
| 11   | ROCK      | ROCK: A robust clustering algorithm for categorical attributes | 2926     | TRUE                  | pyclustering.cluster.rock.rock |
| 12   | GA        | Genetic K-means algorithm | 1851     | TRUE                  | pyclustering.cluster.ga.ga |
| 13   | HDBSCAN   | hdbscan: Hierarchical density based clustering | 1607     | TRUE                  | sklearn.cluster.HDBSCAN |
| 14   | CLARANS   | CLARANS: A method for clustering objects for spatial data mining | 1531     | TRUE                  | pyclustering.cluster.clarans.clarans |


## Directory Structure

### `notebooks/`

**Purpose**: 
- For exploratory data analysis, visualizations, and presenting methodologies in an interactive format.

**Content**: 
- Jupyter notebooks demonstrating the application of functions and classes from the `src` directory. 
- Each notebook is dedicated to a particular part of the analysis or a specific clustering algorithm.

**Reproducibility**: 
- Serves as an interactive report, showcasing findings and insights in a format accessible to those not involved in the codebase's development.

### `src/`

**Purpose**: 
- Houses the core Python code, including functions, classes, and constants – the backbone of the project.

**Content**: 
- Modular, reusable code for data preprocessing, algorithm implementations, and evaluation metrics.
- This code is essential for the functionality within the Jupyter notebooks.

**Maintainability**: 
- Facilitates easier maintenance, testing, and reuse of code. 
- Tailored for 'behind-the-scenes' tasks like algorithm optimization and code refactoring.

### Importance of Separation:
- **Separation of Concerns**: This structure maintains a clear distinction between interactive analysis (`notebooks`) and core code functionality (`src`).
- **Reusability and Scalability**: Promotes reusability of code and scalability of the project. As the project grows, this separation ensures manageability.
- **Collaboration-Friendly**: Addresses version control challenges with notebooks and allows for more traditional software development practices in `src`.

### `results/figures/`

**Purpose**: 
- Dedicated space for storing visual outputs like charts, graphs, and plots.

**Benefits**:
- Centralizes all visualizations for easy access and reference.
- Facilitates integration of visuals into reports and presentations.

### `results/metrics/`

**Purpose**: 
- For storing quantitative evaluation results, such as performance metrics of different algorithms.

**Benefits**:
- Centralizes performance metrics for comparative analysis and historical tracking.
- Aids in the aggregation and analysis of performance data.

## Getting Started
To use this repository:

1. Clone the repo to your local machine.
2. Install the required dependencies listed in `requirements.txt`.
3. Explore the `notebooks` for interactive analysis and visualization.
4. Dive into the `src` directory for an in-depth look at the underlying code.

## Contributing
Contributions to this project are welcome. Please refer to the `CONTRIBUTING.md` file for guidelines.

## License
This project is licensed under the terms of the MIT license.

## Contact
For any queries or feedback related to this project, please open an issue in the repository.
