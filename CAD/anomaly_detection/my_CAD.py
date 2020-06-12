import numpy as np
from sklearn.neighbors import kneighbors_graph as KNN   # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cluster import KMeans
import time


class My_CAD:

    def __init__(self, n_neighbors=10, n_components=None, anomalyDetection=True, kernel="rbf"):
        # X: rows are features and columns are samples
        self.n_samples = None
        self.n_neighbors = n_neighbors
        self.scores = None
        self.scores_outOfSample = None
        self.threshold = None
        self.X = None
        self.kernel = kernel
        self.the_kernel = None
        if self.kernel == "linear" or self.kernel == "cosine" or self.kernel == "sigmoid":
            self.inverse_approach = True
        else:
            self.inverse_approach = False
        # self.inverse_approach = False
        self.y_pred = None
        self.anomalyDetection = anomalyDetection
        self.kmeans_clustering = None
        self.which_cluster_is_anomaly = None

    def CAD_fit(self, X, display_scree_plot=False):
        self.X = X
        self.n_samples = X.shape[1]
        self.find_KNN_inInputSpace(X=X)
        self.scores = self.ranking_score(X=X, outOfSample=False)
        if self.anomalyDetection:
            idx_of_sort = self.scores.argsort()[::-1]  # sort descending
            scores_sorted = self.scores[idx_of_sort]
            self.find_the_threshold_by_clustering(kernel_approach=False)
            if display_scree_plot:
                plt.bar(height=scores_sorted, x=np.arange(1,len(self.scores)+1), align='center', alpha=0.5)
                plt.show()
                plt.bar(height=self.scores, x=np.arange(1, len(self.scores) + 1), align='center', alpha=0.5)
                plt.show()

    def kernel_CAD_fit(self, X, display_scree_plot=False):
        self.X = X
        self.n_samples = X.shape[1]
        self.find_KNN_inFeatureSpace(X=X)
        # self.find_KNN_inInputSpace(X=X)
        self.scores = self.kernel_ranking_score(X=X, outOfSample=False)
        if self.anomalyDetection:
            idx_of_sort = self.scores.argsort()[::-1]  # sort descending
            scores_sorted = self.scores[idx_of_sort]
            self.find_the_threshold_by_clustering(kernel_approach=True)
            if display_scree_plot:
                plt.bar(height=scores_sorted, x=np.arange(1,len(self.scores)+1), align='center', alpha=0.5)
                plt.show()
                plt.bar(height=self.scores, x=np.arange(1, len(self.scores) + 1), align='center', alpha=0.5)
                plt.show()

    def find_anomaly_paths(self, n_iterations, version=1, eta=0.001, termination_check=True):
        # eta = 0.001
        neighbor_indices_fromNormalPoints = self.find_KNN_AnomalyTrack_inInputSpace(X=self.X, exclude_anomalies_from_neighbors=True)
        n_dimensions = self.X.shape[0]
        path_coordinates = np.zeros((self.n_samples, n_dimensions, n_iterations))
        for sample_index in range(self.n_samples):
            if self.y_pred[sample_index] != -1: #--> if not anomaly
                continue
            print("Sample " + str(sample_index) + "/" + str(self.n_samples) + "...")
            sample_x = self.X[:, sample_index].reshape((-1, 1))
            X_updated = self.X.copy()
            cost = self.CPE_cost(X=X_updated, sample_index=sample_index, neighbor_indices=neighbor_indices_fromNormalPoints)
            # print("====== cost: " + str(cost))
            costs = np.zeros((n_iterations,))
            for iteration in range(n_iterations):
                # start = time.time()
                if version == 2:
                    neighbor_indices_fromNormalPoints = self.find_KNN_AnomalyTrack_inInputSpace(X=X_updated, exclude_anomalies_from_neighbors=True)
                gradient = self.CPE_cost_gradient(sample_index=sample_index, sample_x=sample_x, X=self.X, neighbor_indices=neighbor_indices_fromNormalPoints)
                # print(gradient)
                sample_x = (sample_x - (eta * gradient))
                X_updated[:, sample_index] = sample_x.ravel()
                path_coordinates[sample_index, :, iteration] = sample_x.ravel()
                # end = time.time()
                # time_of_an_iteration = end - start
                # print(time_of_an_iteration)
                # input("hiii")
                cost = self.CPE_cost(X=X_updated, sample_index=sample_index, neighbor_indices=neighbor_indices_fromNormalPoints)
                print("====== iteration: " + str(iteration) + ", cost: " + str(cost))
                costs[iteration] = cost
                if termination_check:
                    if iteration >= 1:
                        if costs[iteration] > costs[iteration-1]:
                            path_coordinates[sample_index, :, iteration + 1:] = np.tile(sample_x, (1, n_iterations - iteration - 1))
                            break
                # window_length = 100
                # if iteration >= window_length:
                #     costs_in_window = costs[(iteration-window_length):iteration]
                #     if np.std(costs_in_window) < 0.01:
                #         path_coordinates[sample_index, :, iteration+1:] = np.tile(sample_x, (1, n_iterations-iteration-1))
                #         break
            # input("hi")
        return path_coordinates, self.y_pred

    def CPE_cost(self, X, sample_index, neighbor_indices):
        # neighbor_indices = self.neighbor_indices
        n_samples = X.shape[1]
        sample_x = X[:, sample_index].reshape((-1, 1))
        the_sum = 0
        for neighbor1_index in range(self.n_neighbors - 1):
            the_index = neighbor_indices[sample_index, neighbor1_index].reshape((-1, 1))
            the_index = int(the_index)
            x_a = X[:, the_index].reshape((-1, 1))
            x_a_hat = x_a - sample_x
            for neighbor2_index in range(neighbor1_index, self.n_neighbors):
                the_index = neighbor_indices[sample_index, neighbor2_index].reshape((-1, 1))
                the_index = int(the_index)
                x_b = X[:, the_index].reshape((-1, 1))
                x_b_hat = x_b - sample_x
                cosine_x = self.cosine_of_vectors(x1=x_a_hat, x2=x_b_hat)
                the_sum = the_sum + cosine_x
        cost = the_sum
        return cost

    def CPE_cost_gradient(self, sample_index, sample_x, X, neighbor_indices):
        the_sum = 0
        for neighbor1_index in range(self.n_neighbors - 1):
            the_index = neighbor_indices[sample_index, neighbor1_index].reshape((-1, 1))
            the_index = int(the_index)
            x_a = X[:, the_index].reshape((-1, 1))
            x_a_hat = x_a - sample_x
            for neighbor2_index in range(neighbor1_index, self.n_neighbors):
                the_index = neighbor_indices[sample_index, neighbor2_index].reshape((-1, 1))
                the_index = int(the_index)
                x_b = X[:, the_index].reshape((-1, 1))
                x_b_hat = x_b - sample_x
                # cosine_x = self.cosine_of_vectors(x1=x_a_hat, x2=x_b_hat)
                temp2 = LA.norm(x_a_hat, 2) * LA.norm(x_b_hat, 2)
                temp3 = 1.0 / temp2
                temp4 = -1 * (x_a_hat + x_b_hat)
                temp5 = (x_a_hat.T).dot(x_b_hat)
                temp6 = (1 / ((LA.norm(x_a_hat, 2)) ** 2)) * x_a_hat
                temp7 = (1 / ((LA.norm(x_b_hat, 2)) ** 2)) * x_b_hat
                temp8 = temp3 * (temp4 + (temp5 * (temp6 + temp7)))
                the_sum = the_sum + (temp8)
        gradient = the_sum
        return gradient



    # def CAD_predict(self, X):
    #     # y_pred: -1 for anomaly and 1 for normal
    #     n_samples = X.shape[1]
    #     y_pred = np.zeros((n_samples,))
    #     mask = (self.scores >= self.threshold)
    #     y_pred[mask == 1] = -1
    #     y_pred[mask == 0] = 1
    #     y_pred = y_pred.astype(int)
    #     return y_pred, self.scores

    def CAD_predict(self, X):
        # y_pred: -1 for anomaly and 1 for normal
        return self.y_pred, self.scores

    def set_y_pred(self, forced_y_pred, X):
        self.y_pred = forced_y_pred
        self.n_samples = X.shape[1]
        self.X = X

    def CAD_predict_outOfSample(self, X, exclude_anomalies_from_neighbors):
        self.find_KNN_inInputSpace_for_outOfSample(data_outOfSample=X, calculate_again=True, exclude_anomalies_from_neighbors=exclude_anomalies_from_neighbors)
        self.scores_outOfSample = self.ranking_score(X=X, outOfSample=True)
        # print(list(self.scores_outOfSample))
        y_pred_kmeans = self.kmeans_clustering.predict(X=self.scores_outOfSample.reshape((-1, 1)))
        # kmeans = KMeans(n_clusters=2, random_state=0)
        # Xmm = self.scores_outOfSample.copy()
        # Xmm = Xmm.reshape((-1, 1))
        # y_pred_kmeans = kmeans.fit_predict(X=Xmm)
        # print(list(y_pred_kmeans))
        # plt.bar(height=self.scores_outOfSample, x=np.arange(1, len(self.scores_outOfSample) + 1), align='center', alpha=0.5)
        # plt.show()
        # input("hiiii")
        y_pred = np.zeros((X.shape[1]))
        y_pred[y_pred_kmeans == self.which_cluster_is_anomaly] = -1
        y_pred[y_pred_kmeans != self.which_cluster_is_anomaly] = 1
        y_pred = y_pred.astype(int)
        # print(list(y_pred))
        # plt.bar(height=y_pred, x=np.arange(1, len(self.scores_outOfSample) + 1), align='center',
        #         alpha=0.5)
        # plt.show()
        # print(list(self.scores_outOfSample))
        # input("hiii")
        return y_pred, self.scores_outOfSample

    def kernel_CAD_predict_outOfSample(self, X, exclude_anomalies_from_neighbors):
        self.find_KNN_inFeatureSpace_for_outOfSample(data_outOfSample=X, calculate_again=True, exclude_anomalies_from_neighbors=exclude_anomalies_from_neighbors)
        # self.find_KNN_inInputSpace_for_outOfSample(data_outOfSample=X, calculate_again=True, exclude_anomalies_from_neighbors=False)
        self.scores_outOfSample = self.kernel_ranking_score(X=X, outOfSample=True)
        # print(list(self.scores_outOfSample))
        y_pred_kmeans = self.kmeans_clustering.predict(X=self.scores_outOfSample.reshape((-1, 1)))
        y_pred = np.zeros((X.shape[1]))
        y_pred[y_pred_kmeans == self.which_cluster_is_anomaly] = -1
        y_pred[y_pred_kmeans != self.which_cluster_is_anomaly] = 1
        y_pred = y_pred.astype(int)
        return y_pred, self.scores_outOfSample

    def find_the_threshold_by_clustering(self, kernel_approach=False):
        kmeans = KMeans(n_clusters=2, random_state=0)
        X = self.scores.copy()
        X = X.reshape((-1, 1))
        y_pred_kmeans = kmeans.fit_predict(X=X)
        X = X.ravel()
        scores_cluster1 = X[y_pred_kmeans == 0]
        scores_cluster2 = X[y_pred_kmeans == 1]
        mean_scores_cluster1 = np.mean(scores_cluster1)
        mean_scores_cluster2 = np.mean(scores_cluster2)
        y_pred = np.zeros((self.n_samples,))
        if kernel_approach:
            if not self.inverse_approach:
                if mean_scores_cluster1 <= mean_scores_cluster2:
                    self.which_cluster_is_anomaly = 0
                    y_pred[y_pred_kmeans == 0] = -1 #--> anomaly
                    y_pred[y_pred_kmeans == 1] = 1
                else:
                    self.which_cluster_is_anomaly = 1
                    y_pred[y_pred_kmeans == 0] = 1
                    y_pred[y_pred_kmeans == 1] = -1 # --> anomaly
            else:
                if mean_scores_cluster1 <= mean_scores_cluster2:
                    self.which_cluster_is_anomaly = 1
                    y_pred[y_pred_kmeans == 0] = 1
                    y_pred[y_pred_kmeans == 1] = -1 #--> anomaly
                else:
                    self.which_cluster_is_anomaly = 0
                    y_pred[y_pred_kmeans == 0] = -1 # --> anomaly
                    y_pred[y_pred_kmeans == 1] = 1
        else:
            if mean_scores_cluster1 <= mean_scores_cluster2:
                self.which_cluster_is_anomaly = 1
                y_pred[y_pred_kmeans == 0] = 1
                y_pred[y_pred_kmeans == 1] = -1  # --> anomaly
            else:
                self.which_cluster_is_anomaly = 0
                y_pred[y_pred_kmeans == 0] = -1  # --> anomaly
                y_pred[y_pred_kmeans == 1] = 1
        self.y_pred = y_pred.astype(int)
        self.kmeans_clustering = kmeans

    def find_KNN_inInputSpace_for_outOfSample(self, data_outOfSample, calculate_again=True, exclude_anomalies_from_neighbors=True):
        # data_outOfSample --> rows: features, columns: samples
        n_testing_images = data_outOfSample.shape[1]
        if calculate_again:
            self.neighbor_indices_for_outOfSample = np.zeros((n_testing_images, self.n_neighbors))
            # --- KNN:
            for test_sample_index in range(n_testing_images):
                test_sample = data_outOfSample[:, test_sample_index].reshape((-1, 1))
                distances_from_this_outOfSample_image = np.zeros((self.n_samples,))
                for train_sample_index in range(self.n_samples):
                    if exclude_anomalies_from_neighbors:
                        if self.y_pred[train_sample_index] == -1:  # --> if anomaly
                            distances_from_this_outOfSample_image[train_sample_index] = np.inf
                        else:
                            train_sample = self.X[:, train_sample_index].reshape((-1, 1))
                            distances_from_this_outOfSample_image[train_sample_index] = LA.norm(test_sample - train_sample)
                    else:
                        train_sample = self.X[:, train_sample_index].reshape((-1, 1))
                        distances_from_this_outOfSample_image[train_sample_index] = LA.norm(test_sample - train_sample)
                argsort_distances = np.argsort(distances_from_this_outOfSample_image.ravel())  # arg of ascending sort
                indices_of_neighbors_of_this_outOfSample_image = argsort_distances[:self.n_neighbors+1] #--> remove itself as its neighbor
                self.neighbor_indices_for_outOfSample[test_sample_index, :] = indices_of_neighbors_of_this_outOfSample_image[1:] #--> remove itself as its neighbor
            # --- save KNN:
            # self.save_variable(variable=self.neighbor_indices_for_outOfSample, name_of_variable="neighbor_indices_for_outOfSample", path_to_save="./LLISE_settings/LLISE/")
        else:
            pass
            # self.neighbor_indices_for_outOfSample = self.load_variable(name_of_variable="neighbor_indices_for_outOfSample", path="./LLISE_settings/LLISE/")

    def find_KNN_inFeatureSpace_for_outOfSample(self, data_outOfSample, calculate_again=True, exclude_anomalies_from_neighbors=True):
        # data_outOfSample --> rows: features, columns: samples
        the_kernel_outOfSample_and_training = pairwise_kernels(X=data_outOfSample.T, Y=(self.X).T, metric=self.kernel)
        # the_kernel_outOfSample_and_training = pairwise_kernels(X=data_outOfSample.T, Y=(self.X).T, metric=self.kernel, degree=10)
        the_kernel_outOfSample_and_outOfSample = pairwise_kernels(X=data_outOfSample.T, Y=data_outOfSample.T, metric=self.kernel)
        # the_kernel_outOfSample_and_outOfSample = pairwise_kernels(X=data_outOfSample.T, Y=data_outOfSample.T, metric=self.kernel, degree=10)
        n_testing_images = data_outOfSample.shape[1]
        if calculate_again:
            self.neighbor_indices_for_outOfSample = np.zeros((n_testing_images, self.n_neighbors))
            # --- KNN:
            for test_sample_index in range(n_testing_images):
                distances_from_this_outOfSample_image = np.zeros((self.n_samples,))
                for train_sample_index in range(self.n_samples):
                    if exclude_anomalies_from_neighbors:
                        if self.y_pred[train_sample_index] == -1:  # --> if anomaly
                            distances_from_this_outOfSample_image[train_sample_index] = np.inf
                        else:
                            temp1 = the_kernel_outOfSample_and_outOfSample[test_sample_index, test_sample_index]
                            temp2 = self.the_kernel[train_sample_index, train_sample_index]
                            temp3 = the_kernel_outOfSample_and_training[test_sample_index, train_sample_index]
                            temp = temp1 - 2 * temp3 + temp2
                            if temp < 0:
                                # might occur for imperfect software calculations
                                temp = 0
                            distance = temp ** 0.5
                            distances_from_this_outOfSample_image[train_sample_index] = distance
                    else:
                        temp1 = the_kernel_outOfSample_and_outOfSample[test_sample_index, test_sample_index]
                        temp2 = self.the_kernel[train_sample_index, train_sample_index]
                        temp3 = the_kernel_outOfSample_and_training[test_sample_index, train_sample_index]
                        temp = temp1 - 2 * temp3 + temp2
                        if temp < 0:
                            # might occur for imperfect software calculations
                            temp = 0
                        distance = temp ** 0.5
                        distances_from_this_outOfSample_image[train_sample_index] = distance
                argsort_distances = np.argsort(distances_from_this_outOfSample_image.ravel())  # arg of ascending sort
                indices_of_neighbors_of_this_outOfSample_image = argsort_distances[:self.n_neighbors+1] #--> remove itself as its neighbor
                self.neighbor_indices_for_outOfSample[test_sample_index, :] = indices_of_neighbors_of_this_outOfSample_image[1:] #--> remove itself as its neighbor
            # --- save KNN:
            # self.save_variable(variable=self.neighbor_indices_for_outOfSample, name_of_variable="neighbor_indices_for_outOfSample", path_to_save="./LLISE_settings/LLISE/")
        else:
            pass
            # self.neighbor_indices_for_outOfSample = self.load_variable(name_of_variable="neighbor_indices_for_outOfSample", path="./LLISE_settings/LLISE/")

    def find_KNN_inInputSpace(self, X):
        self.neighbor_indices = np.zeros((self.n_samples, self.n_neighbors))
        # --- KNN:
        connectivity_matrix = KNN(X=X.T, n_neighbors=self.n_neighbors, mode='connectivity', include_self=False, n_jobs=-1)
        connectivity_matrix = connectivity_matrix.toarray()
        # --- store indices of neighbors:
        for sample_index in range(self.n_samples):
            self.neighbor_indices[sample_index, :] = np.argwhere(connectivity_matrix[sample_index, :] == 1).ravel()

    def find_KNN_AnomalyTrack_inInputSpace(self, X, exclude_anomalies_from_neighbors=True):
        # X_anomaly = X[:, self.y_pred == -1]
        # X_normal = X[:, self.y_pred == 1]
        neighbor_indices = np.zeros((self.n_samples, self.n_neighbors))
        for sample_index_1 in range(self.n_samples):
            if self.y_pred[sample_index_1] != -1:  # --> if not anomaly
                neighbor_indices[sample_index_1, :] = -1 * np.ones((self.n_neighbors,))  # --> invalid
            else:
                sample1 = X[:, sample_index_1]
                distances_from_this_instance = np.zeros((1, self.n_samples))
                for sample_index_2 in range(self.n_samples):
                    if exclude_anomalies_from_neighbors:
                        if self.y_pred[sample_index_2] == -1:  # --> if anomaly
                            distances_from_this_instance[0, sample_index_2] = np.inf
                        else:
                            sample2 = X[:, sample_index_2]
                            distances_from_this_instance[0, sample_index_2] = LA.norm(sample1 - sample2)
                    else:
                        sample2 = X[:, sample_index_2]
                        distances_from_this_instance[0, sample_index_2] = LA.norm(sample1 - sample2)
                argsort_distances = np.argsort(distances_from_this_instance.ravel())  # arg of ascending sort
                indices_of_neighbors_of_this_image = argsort_distances[:self.n_neighbors+1] #--> remove itself as its neighbor
                neighbor_indices[sample_index_1, :] = indices_of_neighbors_of_this_image[1:] #--> remove itself as its neighbor
        return neighbor_indices

    def ranking_score(self, X, outOfSample=False):
        if outOfSample:
            neighbor_indices = self.neighbor_indices_for_outOfSample
        else:
            neighbor_indices = self.neighbor_indices
        n_samples = X.shape[1]
        scores = np.zeros((n_samples,))
        for sample_index in range(n_samples):
            sample = X[:, sample_index].reshape((-1, 1))
            the_sum = 0
            for neighbor1_index in range(self.n_neighbors - 1):
                the_index = neighbor_indices[sample_index, neighbor1_index].reshape((-1, 1))
                the_index = int(the_index)
                # x_a = X[:, the_index].reshape((-1, 1))
                x_a = self.X[:, the_index].reshape((-1, 1))
                x_a_hat = x_a - sample
                for neighbor2_index in range(neighbor1_index, self.n_neighbors):
                    the_index = neighbor_indices[sample_index, neighbor2_index].reshape((-1, 1))
                    the_index = int(the_index)
                    # x_b = X[:, the_index].reshape((-1, 1))
                    x_b = self.X[:, the_index].reshape((-1, 1))
                    x_b_hat = x_b - sample
                    cosine = self.cosine_of_vectors(x1=x_a_hat, x2=x_b_hat)
                    the_sum = the_sum + cosine
            scores[sample_index] = the_sum
        # if not self.anomalyDetection:
        #     # if not self.inverse_approach:
        #     scores = -1 * scores
        return scores

    def kernel_ranking_score(self, X, outOfSample=False):
        if outOfSample:
            neighbor_indices = self.neighbor_indices_for_outOfSample
        else:
            neighbor_indices = self.neighbor_indices
        n_samples = X.shape[1]
        scores = np.zeros((n_samples,))
        for sample_index in range(n_samples):
            sample = X[:, sample_index].reshape((-1, 1))
            the_indices = neighbor_indices[sample_index, :]
            the_indices = the_indices.astype(int)
            # X_neighbors = X[:, the_indices]
            X_neighbors = self.X[:, the_indices]
            kernel_cosine_matrix = self.kernel_cosine_of_vectors(x=sample, X_neighbors=X_neighbors)
            the_sum = 0
            for neighbor1_index in range(self.n_neighbors - 1):
                for neighbor2_index in range(neighbor1_index, self.n_neighbors):
                    kernel_cosine = kernel_cosine_matrix[neighbor1_index, neighbor2_index]
                    the_sum = the_sum + kernel_cosine
            scores[sample_index] = the_sum
        # if not self.anomalyDetection:
        #     scores = -1 * scores
        return scores

    def kernel_cosine_of_vectors(self, x, X_neighbors):
        # x: column vector
        # X_neighbors: columns are samples, rows are features
        X_neighbors = X_neighbors - x
        kernel_over_neighbors = pairwise_kernels(X=X_neighbors.T, Y=X_neighbors.T, metric=self.kernel)
        # kernel_over_neighbors = pairwise_kernels(X=X_neighbors.T, Y=X_neighbors.T, metric=self.kernel, degree=10)
        kernel_cosine_matrix = self.normalize_the_kernel(kernel_matrix=kernel_over_neighbors)
        return kernel_cosine_matrix

    def cosine_of_vectors(self, x1, x2):
        # x1, x2: column vectors
        numerator = (x1.T).dot(x2)
        numerator = numerator.ravel()[0]
        denominator = LA.norm(x1, 2) * LA.norm(x2, 2)
        if numerator == 0 and denominator == 0:
            cosine = 0
        else:
            cosine = numerator / denominator
        if cosine > 1:  # because of computer imperfect calculations
            cosine = 1
        return cosine

    def normalize_the_kernel(self, kernel_matrix):
        diag_kernel = np.diag(kernel_matrix)
        k = (1 / np.sqrt(diag_kernel)).reshape((-1, 1))
        normalized_kernel_matrix = np.multiply(kernel_matrix, k.dot(k.T))
        return normalized_kernel_matrix

    def find_KNN_inFeatureSpace(self, X):
        self.neighbor_indices = np.zeros((self.n_samples, self.n_neighbors))
        # --- kernel:
        self.the_kernel = pairwise_kernels(X=X.T, Y=X.T, metric=self.kernel)
        # self.the_kernel = pairwise_kernels(X=X.T, Y=X.T, metric=self.kernel, degree=10)
        # --- KNN:
        for sample_index_1 in range(self.n_samples):
            distances_from_this_instance = np.zeros((1, self.n_samples))
            for sample_index_2 in range(self.n_samples):
                distances_from_this_instance[0, sample_index_2] = self.distance_based_on_kernel(kernel_matrix=self.the_kernel, index1=sample_index_1, index2=sample_index_2)
            argsort_distances = np.argsort(distances_from_this_instance.ravel())  # arg of ascending sort
            indices_of_neighbors_of_this_image = argsort_distances[:self.n_neighbors+1] #--> remove itself as its neighbor
            self.neighbor_indices[sample_index_1, :] = indices_of_neighbors_of_this_image[1:] #--> remove itself as its neighbor

    def distance_based_on_kernel(self, kernel_matrix, index1, index2):
        temp = kernel_matrix[index1, index1] - 2 * kernel_matrix[index1, index2] + kernel_matrix[index2, index2]
        if temp < 0:
            # might occur for imperfect software calculations
            temp = 0
        distance = temp ** 0.5
        return distance

    ############### functions for numerosity reduction:

    def CAD_rank_instances(self, scores, kernel_approach, display_plot=False):
        # convert anomaly scores to numerosity (inlier) scores:
        if kernel_approach:
            if self.inverse_approach:
                scores = -1 * scores
        else:
            scores = -1 * scores
        # sort according to scores:
        idx_of_sort = scores.argsort()[::-1]  # sort descending
        rank = idx_of_sort
        if display_plot:
            plt.bar(height=scores[idx_of_sort], x=np.arange(1, len(scores) + 1), align='center', alpha=0.5)
            plt.show()
        # print(rank)
        # print(scores[rank[0]])
        # print(scores[rank[1]])
        # print(scores[rank[-1]])
        return rank

    def sort_samples(self, scores, X, Y, kernel_approach):
        # output: X_sorted, Y_sorted, scores_sorted --> (rows are features, columns are samples)
        n_dimensions = X.shape[0]
        # convert anomaly scores to numerosity (inlier) scores:
        if kernel_approach:
            if self.inverse_approach:
                scores = -1 * scores
        else:
            scores = -1 * scores
        if Y is None:
            X_with_scores = np.vstack((scores, X))
            # sort matrix with respect to values in first row:
            X_with_scores_sorted = self.sort_matrix(X=X_with_scores, withRespectTo_columnOrRow='row', index_columnOrRow=0, descending=True)
            X_sorted = X_with_scores_sorted[1:, :]
            scores_sorted = X_with_scores_sorted[0, :]
            Y_sorted = None
        else:
            X_with_scores = np.vstack((scores, X))
            X_with_scores_and_Y = np.vstack((X_with_scores, Y))
            # sort matrix with respect to values in first row:
            X_with_scores_and_Y_sorted = self.sort_matrix(X=X_with_scores_and_Y, withRespectTo_columnOrRow='row', index_columnOrRow=0, descending=True)
            X_sorted = X_with_scores_and_Y_sorted[1:1 + n_dimensions, :]
            Y_sorted = X_with_scores_and_Y_sorted[n_dimensions + 1:, :]
            scores_sorted = X_with_scores_and_Y_sorted[0, :]
        return X_sorted, Y_sorted, scores_sorted

    def sort_matrix(self, X, withRespectTo_columnOrRow='column', index_columnOrRow=0, descending=True):
        # I googled: python sort matrix by first row --> https://gist.github.com/stevenvo/e3dad127598842459b68
        # https://stackoverflow.com/questions/37856290/python-argsort-in-descending-order-for-2d-array
        # sort array with regards to nth column or row:
        if withRespectTo_columnOrRow == 'column':
            if descending is True:
                X = X[X[:, index_columnOrRow].argsort()][::-1]
            else:
                X = X[X[:, index_columnOrRow].argsort()]
        elif withRespectTo_columnOrRow == 'row':
            X = X.T
            if descending is True:
                X = X[X[:, index_columnOrRow].argsort()][::-1]
            else:
                X = X[X[:, index_columnOrRow].argsort()]
            X = X.T
        return X

    def CAD_prototype_selection(self, scores, kernel_approach):
        # convert anomaly scores to numerosity (inlier) scores:
        if kernel_approach:
            if self.inverse_approach:
                scores = -1 * scores
        else:
            scores = -1 * scores
        # prototype selection:
        kmeans = KMeans(n_clusters=2, random_state=0)
        scores = scores.reshape((-1, 1))
        y_pred_kmeans = kmeans.fit_predict(X=scores)
        scores_cluster1 = scores[y_pred_kmeans == 0]
        scores_cluster2 = scores[y_pred_kmeans == 1]
        mean_scores_cluster1 = np.mean(scores_cluster1)
        mean_scores_cluster2 = np.mean(scores_cluster2)
        kept_prototypes_indices = np.zeros((self.n_samples,))
        if mean_scores_cluster1 <= mean_scores_cluster2:
            kept_prototypes_indices[y_pred_kmeans == 0] = 0
            kept_prototypes_indices[y_pred_kmeans == 1] = 1 #--> keep these
        else:
            kept_prototypes_indices[y_pred_kmeans == 0] = 1 # --> keep these
            kept_prototypes_indices[y_pred_kmeans == 1] = 0
        kept_prototypes_indices = kept_prototypes_indices.reshape((1, -1))
        return kept_prototypes_indices