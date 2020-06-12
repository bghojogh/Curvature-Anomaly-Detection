import numpy as np
from sklearn.neighbors import NearestNeighbors as KNN  # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


# Sample Reduction with DROP:
# Paper: Reduction Techniques for Instance-Based Learning Algorithms
class SR_Drop:

    def __init__(self, X, Y, n_neighbors):
        # X: rows are features and columns are samples
        # Y: rows are features and columns are samples
        self.X = X
        self.Y = Y
        self.n_dimensions = X.shape[0]
        self.n_samples = X.shape[1]
        self.n_neighbors = n_neighbors

    def Drop1_prototype_selection(self):
        last_index_removal_in_X = None
        # --- find k-nearest neighbor graph (distance matrix):
        knn = KNN(n_neighbors=self.n_samples, algorithm='kd_tree')
        knn.fit(X=(self.X).T)
        distance_matrix = knn.kneighbors_graph(X=(self.X).T, n_neighbors=self.n_samples, mode='distance')
        distance_matrix = distance_matrix.toarray()
        kept_prototypes_indices = np.ones((1, self.n_samples))
        # process every point (whether to remove it or not):
        for sample_index in range(self.n_samples):
            connectivity_matrix = np.zeros((self.n_samples, self.n_samples))
            # --- remove the removed sample from KNN graph:
            if last_index_removal_in_X is not None:
                distance_matrix[:, last_index_removal_in_X] = np.inf  # set to inf so when sorting, it will be last (not neighbor to any point)
                distance_matrix[last_index_removal_in_X, :] = np.inf  # set to inf so when sorting, it will be last (not neighbor to any point)
            # --- find (update again) k-nearest neighbors of every sample:
            for sample_index_2 in range(self.n_samples):
                distances_from_neighbors = distance_matrix[sample_index_2, :]
                sorted_neighbors_by_distance = distances_from_neighbors.argsort()  # ascending order
                n_neighbors = min(self.n_neighbors, np.sum(distances_from_neighbors != np.inf))  # in last iterations, the number of left samples becomes less than self.n_neighbors
                for neighbor_index in range(n_neighbors):
                    index_of_neighbor = sorted_neighbors_by_distance[neighbor_index]
                    connectivity_matrix[sample_index_2, index_of_neighbor] = 1
            # --- replace zeros with nan:
            connectivity_matrix[connectivity_matrix == 0] = np.nan
            # --- replace ones (connectivities) with labels:
            labels = self.Y.reshape((1, -1))
            repeated_labels_in_rows = np.tile(labels, (self.n_samples, 1))
            connectivity_matrix_having_labels = np.multiply(connectivity_matrix, repeated_labels_in_rows)
            # --- identifying neighbors of sample (using connectivity matrix): --> identifying points which have this sample as their associate
            indices_of_neighbors_of_that_sample = [i for i in range(self.n_samples) if ~(np.isnan(connectivity_matrix_having_labels[i, sample_index]))]
            # --- with the sample:
            classified_correctly_withSample = 0
            for neighbor_sample_index in indices_of_neighbors_of_that_sample:
                label_of_neighbor_sample = self.Y[:, neighbor_sample_index]
                n_similar_samples = np.sum(connectivity_matrix_having_labels[neighbor_sample_index, :] == label_of_neighbor_sample) - 1  # we exclude the sample itself from neighbors
                n_dissimilar_samples = np.sum((connectivity_matrix_having_labels[neighbor_sample_index, :] != label_of_neighbor_sample) & ~(np.isnan(connectivity_matrix_having_labels[neighbor_sample_index, :])))
                if n_similar_samples > n_dissimilar_samples:
                    classified_correctly_withSample = classified_correctly_withSample + 1
            # --- without the sample:
            # connectivity_matrix_without_sample = connectivity_matrix_having_labels.copy()
            connectivity_matrix_without_sample = connectivity_matrix_having_labels
            connectivity_matrix_without_sample[:, sample_index] = np.nan
            classified_correctly_withoutSample = 0
            for neighbor_sample_index in indices_of_neighbors_of_that_sample:
                label_of_neighbor_sample = self.Y[:, neighbor_sample_index]
                n_similar_samples = np.sum(connectivity_matrix_without_sample[neighbor_sample_index, :] == label_of_neighbor_sample) - 1  # we exclude the sample itself from neighbors
                n_dissimilar_samples = np.sum((connectivity_matrix_without_sample[neighbor_sample_index, :] != label_of_neighbor_sample) & ~(np.isnan(connectivity_matrix_having_labels[neighbor_sample_index, :])))
                if n_similar_samples > n_dissimilar_samples:
                    classified_correctly_withoutSample = classified_correctly_withoutSample + 1
            # --- check whether to remove sample or not:
            if classified_correctly_withoutSample >= classified_correctly_withSample:
                # should be removed
                last_index_removal_in_X = sample_index
                kept_prototypes_indices[:, sample_index] = 0
        return kept_prototypes_indices

    def Drop2_prototype_selection(self):
        last_index_removal_in_X = None
        # --- find k-nearest neighbor graph (distance matrix):
        knn = KNN(n_neighbors=self.n_samples, algorithm='kd_tree')
        knn.fit(X=(self.X).T)
        distance_matrix = knn.kneighbors_graph(X=(self.X).T, n_neighbors=self.n_samples, mode='distance')
        distance_matrix = distance_matrix.toarray()
        # --- find distance of nearest enemy to every point:
        distance_matrix_copy = distance_matrix.copy()
        labels = self.Y.reshape((1, -1))
        for sample_index in range(self.n_samples):
            label = labels.ravel()[sample_index]
            which_points_are_sameClass = (labels.ravel() == label)
            distance_matrix_copy[sample_index, which_points_are_sameClass] = np.nan #--> make distances of friends nan so enemies remain
        distance_to_nearest_enemy = np.zeros((self.n_samples, 1))
        for sample_index in range(self.n_samples):
            enemy_distances_for_the_sample = distance_matrix_copy[sample_index, :]
            sorted_distances = np.sort(enemy_distances_for_the_sample)  # --> sort ascending  #--> the nan ones will be at the end of sorted list
            distance_to_nearest_enemy[sample_index, 0] = sorted_distances[0]
        distance_to_nearest_enemy = distance_to_nearest_enemy.ravel()
        order_of_indices = (-distance_to_nearest_enemy).argsort()  # --> argsort descending (furthest nearest neighbor to closest nearest neighbor)
        # process every point (whether to remove it or not):
        kept_prototypes_indices = np.ones((1, self.n_samples))
        for sample_index in order_of_indices:
            connectivity_matrix = np.zeros((self.n_samples, self.n_samples))
            # --- remove the removed sample from KNN graph:
            if last_index_removal_in_X is not None:
                # in DROP2 this line is commented --> #distance_matrix[:, last_index_removal_in_X] = np.inf  # set to inf so when sorting, it will be last (not neighbor to any point)
                distance_matrix[last_index_removal_in_X, :] = np.inf  # set to inf so when sorting, it will be last (not neighbor to any point)
            # --- find (update again) k-nearest neighbors of every sample:
            for sample_index_2 in range(self.n_samples):
                distances_from_neighbors = distance_matrix[sample_index_2, :]
                sorted_neighbors_by_distance = distances_from_neighbors.argsort()  # ascending order
                n_neighbors = min(self.n_neighbors, np.sum(distances_from_neighbors != np.inf))  # in last iterations, the number of left samples becomes less than self.n_neighbors
                for neighbor_index in range(n_neighbors):
                    index_of_neighbor = sorted_neighbors_by_distance[neighbor_index]
                    connectivity_matrix[sample_index_2, index_of_neighbor] = 1
            # --- replace zeros with nan:
            connectivity_matrix[connectivity_matrix == 0] = np.nan
            # --- replace ones (connectivities) with labels:
            labels = self.Y.reshape((1, -1))
            repeated_labels_in_rows = np.tile(labels, (self.n_samples, 1))
            connectivity_matrix_having_labels = np.multiply(connectivity_matrix, repeated_labels_in_rows)
            # --- identifying neighbors of sample (using connectivity matrix): --> identifying points which have this sample as their associate
            indices_of_neighbors_of_that_sample = [i for i in range(self.n_samples) if ~(np.isnan(connectivity_matrix_having_labels[i, sample_index]))]
            # --- with the sample:
            classified_correctly_withSample = 0
            for neighbor_sample_index in indices_of_neighbors_of_that_sample:
                label_of_neighbor_sample = self.Y[:, neighbor_sample_index]
                n_similar_samples = np.sum(connectivity_matrix_having_labels[neighbor_sample_index, :] == label_of_neighbor_sample) - 1  # we exclude the sample itself from neighbors
                n_dissimilar_samples = np.sum((connectivity_matrix_having_labels[neighbor_sample_index, :] != label_of_neighbor_sample) & ~(np.isnan(connectivity_matrix_having_labels[neighbor_sample_index, :])))
                if n_similar_samples > n_dissimilar_samples:
                    classified_correctly_withSample = classified_correctly_withSample + 1
            # --- without the sample:
            # connectivity_matrix_without_sample = connectivity_matrix_having_labels.copy()
            connectivity_matrix_without_sample = connectivity_matrix_having_labels
            connectivity_matrix_without_sample[:, sample_index] = np.nan
            classified_correctly_withoutSample = 0
            for neighbor_sample_index in indices_of_neighbors_of_that_sample:
                label_of_neighbor_sample = self.Y[:, neighbor_sample_index]
                n_similar_samples = np.sum(connectivity_matrix_without_sample[neighbor_sample_index, :] == label_of_neighbor_sample) - 1  # we exclude the sample itself from neighbors
                n_dissimilar_samples = np.sum((connectivity_matrix_without_sample[neighbor_sample_index, :] != label_of_neighbor_sample) & ~(np.isnan(connectivity_matrix_having_labels[neighbor_sample_index, :])))
                if n_similar_samples > n_dissimilar_samples:
                    classified_correctly_withoutSample = classified_correctly_withoutSample + 1
            # --- check whether to remove sample or not:
            if classified_correctly_withoutSample >= classified_correctly_withSample:
                # should be removed
                last_index_removal_in_X = sample_index
                kept_prototypes_indices[:, sample_index] = 0
        return kept_prototypes_indices

    def Drop3_prototype_selection(self):
        n_samples_backup = self.X.shape[1]
        # --- ENN filter:
        kept_prototypes_indices_1 = self.ENN_prototype_selection()
        kept_prototypes_indices_1 = kept_prototypes_indices_1.ravel().astype(int)
        self.X = self.X[:, kept_prototypes_indices_1 == 1]
        self.Y = self.Y[:, kept_prototypes_indices_1 == 1]
        self.n_samples = self.X.shape[1]
        # --- DROP2:
        kept_prototypes_indices_2 = self.Drop2_prototype_selection()
        # --- convert kept_prototypes_indices_2 to kept_prototypes_indices:
        kept_prototypes_indices_1 = kept_prototypes_indices_1.reshape((1, -1))
        kept_prototypes_indices = np.zeros((1, n_samples_backup))
        pivot = -1
        for sample_index in range(n_samples_backup):
            if kept_prototypes_indices_1[:, sample_index] == 1:
                pivot = pivot + 1
                if kept_prototypes_indices_2[:, pivot] == 1:
                    kept_prototypes_indices[:, sample_index] = 1
        return kept_prototypes_indices

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