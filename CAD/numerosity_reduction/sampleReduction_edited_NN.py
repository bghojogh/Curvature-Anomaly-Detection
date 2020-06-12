import numpy as np
from sklearn.neighbors import NearestNeighbors as KNN  # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
import math

# Sample Reduction with edited Nearest Neighbor (ENN):
# Paper: Asymptotic Properties of Nearest Neighbor Rules Using Edited Data
class SR_edited_NN:

    def __init__(self, X, Y, n_neighbors):
        # X: rows are features and columns are samples
        # Y: rows are features and columns are samples
        self.X = X
        self.Y = Y
        self.n_dimensions = X.shape[0]
        self.n_samples = X.shape[1]
        self.n_neighbors = n_neighbors

    def ENN_prototype_selection(self):
        # --- find k-nearest neighbor graph:
        n_neighbors = self.n_neighbors
        knn = KNN(n_neighbors=n_neighbors, algorithm='kd_tree')
        knn.fit(X=(self.X).T)
        connectivity_matrix = knn.kneighbors_graph(X=(self.X).T, n_neighbors=n_neighbors, mode='connectivity')
        connectivity_matrix = connectivity_matrix.toarray()
        # --- replace zeros with nan:
        connectivity_matrix[connectivity_matrix == 0] = np.nan
        # --- replace ones (connectivities) with labels:
        labels = (self.Y).reshape((1, -1))
        repeated_labels_in_rows = np.tile(labels, (self.n_samples, 1))
        connectivity_matrix_having_labels = np.multiply(connectivity_matrix, repeated_labels_in_rows)
        # --- find scores of samples:
        kept_prototypes_indices = np.ones((1, self.n_samples))
        for sample_index in range(self.n_samples):
            label_of_sample = self.Y[:, sample_index]
            n_friends = np.sum(connectivity_matrix_having_labels[sample_index, :] == label_of_sample) - 1  # we exclude the sample itself from neighbors
            n_enemies = np.sum((connectivity_matrix_having_labels[sample_index, :] != label_of_sample) & ~(np.isnan(connectivity_matrix_having_labels[sample_index, :])))
            if n_enemies >= n_friends:
                kept_prototypes_indices[:, sample_index] = 0
        return kept_prototypes_indices

