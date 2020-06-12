import numpy as np
from sklearn.neighbors import kneighbors_graph as KNN   # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import pickle


class My_CPE:

    def __init__(self, n_neighbors=10, n_components=None, kernel="rbf"):
        # X: rows are features and columns are samples
        self.n_samples = None
        self.n_dimensions = None
        self.n_neighbors = n_neighbors
        self.X = None
        self.kernel = kernel
        self.the_kernel = None
        self.neighbor_indices = None
        self.n_components = n_components

    def CPE_fit(self, X, step_checkpoint=10, fit_again=True):
        eta = 0.001
        eta_initial = eta
        if not fit_again:
            path_to_save = './CPE_settings/CPE/eta=' + str(eta_initial) + "/"
            name_of_variable = "Y_8" # "Y_16" #--> iteration 84
            Y = self.load_variable(name_of_variable=name_of_variable, path=path_to_save)
            return Y
        self.n_samples = X.shape[1]
        self.n_dimensions = X.shape[0]
        if self.n_components is None:
            self.n_components = self.n_dimensions
        self.find_KNN_inInputSpace(X)
        Y = np.random.rand(self.n_components, self.n_samples)  # --> rand is in range [0,1)
        # pca = PCA(n_components=self.n_components)
        # Y = (pca.fit_transform(X=X.T)).T
        path_to_save = './CPE_settings/CPE/eta=' + str(eta_initial) + "/"
        self.save_variable(variable=Y, name_of_variable="Y_initial", path_to_save=path_to_save)
        Y_updated = np.zeros((self.n_components, self.n_samples))
        iteration_index = -1
        cost = self.CPE_cost(X=X, Y=Y)
        print("Iteration: " + str(iteration_index) + ", Cost: " + str(cost))
        cost_iters = np.zeros((step_checkpoint, 1))
        eta_iters = np.zeros((step_checkpoint, 1))
        while True:
            cost_previous = cost
            iteration_index = iteration_index + 1
            for sample_index in range(self.n_samples):
                sample_x = X[:, sample_index].reshape((-1, 1))
                sample_y = Y[:, sample_index].reshape((-1, 1))
                gradient = self.CPE_cost_gradient(sample_index=sample_index, sample_x=sample_x, sample_y=sample_y, X=X, Y=Y)
                Y_updated[:, sample_index] = (sample_y - (eta * gradient)).ravel()
                # Y[:, sample_index] = (sample_y - (eta * gradient)).ravel()
            Y = Y_updated.copy()
            # if iteration_index < 50:
            #     for sample_index in range(self.n_samples):
            #         noise = np.random.normal(0, 0.1, self.n_components)
            #         Y[:, sample_index] = Y[:, sample_index] + noise
            cost = self.CPE_cost(X=X, Y=Y)
            print("Iteration: " + str(iteration_index) + ", Cost: " + str(cost))
            # if cost_previous < cost:
            #     eta = 0.9 * eta
            # save the information at checkpoints:
            index_to_save = iteration_index % step_checkpoint
            cost_iters[index_to_save] = cost
            eta_iters[index_to_save] = eta
            if (iteration_index+1) % step_checkpoint == 0:
                path_to_save = './CPE_settings/CPE/eta=' + str(eta_initial) + "/"
                print("Saving the checkpoint in iteration #" + str(iteration_index))
                checkpoint_index = int(np.floor(iteration_index / step_checkpoint))
                self.save_variable(variable=Y, name_of_variable="Y_" + str(checkpoint_index), path_to_save=path_to_save)
                self.save_variable(variable=cost_iters, name_of_variable="cost_iters_" + str(checkpoint_index), path_to_save=path_to_save)
                self.save_np_array_to_txt(variable=cost_iters, name_of_variable="cost_iters_"+str(checkpoint_index), path_to_save=path_to_save)
                self.save_variable(variable=iteration_index, name_of_variable="iteration_index_" + str(checkpoint_index), path_to_save=path_to_save)
                self.save_np_array_to_txt(variable=np.asarray(iteration_index), name_of_variable="iteration_index_"+str(checkpoint_index), path_to_save=path_to_save)
                self.save_variable(variable=eta_iters, name_of_variable="eta_iters_" + str(checkpoint_index), path_to_save=path_to_save)
                self.save_np_array_to_txt(variable=eta_iters, name_of_variable="eta_iters_"+str(checkpoint_index), path_to_save=path_to_save)

    def kernel_CPE_fit(self, X, step_checkpoint=10, fit_again=True):
        eta = 0.001
        eta_initial = eta
        if not fit_again:
            path_to_save = './CPE_settings/kernel_CPE/eta=' + str(eta_initial) + "/"
            name_of_variable = "Y_100" # "Y_16" #--> iteration 84
            Y = self.load_variable(name_of_variable=name_of_variable, path=path_to_save)
            return Y
        self.n_samples = X.shape[1]
        self.n_dimensions = X.shape[0]
        if self.n_components is None:
            self.n_components = self.n_dimensions
        self.find_KNN_inFeatureSpace(X)
        Y = np.random.rand(self.n_components, self.n_samples)  # --> rand is in range [0,1)
        # pca = PCA(n_components=self.n_components)
        # Y = (pca.fit_transform(X=X.T)).T
        path_to_save = './CPE_settings/kernel_CPE/eta=' + str(eta_initial) + "/"
        self.save_variable(variable=Y, name_of_variable="Y_initial", path_to_save=path_to_save)
        Y_updated = np.zeros((self.n_components, self.n_samples))
        iteration_index = -1
        cost = self.kernel_CPE_cost(X=X, Y=Y)
        print("Iteration: " + str(iteration_index) + ", Cost: " + str(cost))
        cost_iters = np.zeros((step_checkpoint, 1))
        eta_iters = np.zeros((step_checkpoint, 1))
        while True:
            cost_previous = cost
            iteration_index = iteration_index + 1
            for sample_index in range(self.n_samples):
                sample_x = X[:, sample_index].reshape((-1, 1))
                sample_y = Y[:, sample_index].reshape((-1, 1))
                gradient = self.kernel_CPE_cost_gradient(sample_index=sample_index, sample_x=sample_x, sample_y=sample_y, X=X, Y=Y)
                Y_updated[:, sample_index] = (sample_y - (eta * gradient)).ravel()
                # Y[:, sample_index] = (sample_y - (eta * gradient)).ravel()
            Y = Y_updated.copy()
            if iteration_index < 50:
                for sample_index in range(self.n_samples):
                    noise = np.random.normal(0, 0.1, self.n_components)
                    Y[:, sample_index] = Y[:, sample_index] + noise
            cost = self.kernel_CPE_cost(X=X, Y=Y)
            print("Iteration: " + str(iteration_index) + ", Cost: " + str(cost))
            # if cost_previous < cost:
            #     eta = 0.9 * eta
            # save the information at checkpoints:
            index_to_save = iteration_index % step_checkpoint
            cost_iters[index_to_save] = cost
            eta_iters[index_to_save] = eta
            if (iteration_index+1) % step_checkpoint == 0:
                path_to_save = './CPE_settings/kernel_CPE/eta=' + str(eta_initial) + "/"
                print("Saving the checkpoint in iteration #" + str(iteration_index))
                checkpoint_index = int(np.floor(iteration_index / step_checkpoint))
                self.save_variable(variable=Y, name_of_variable="Y_" + str(checkpoint_index), path_to_save=path_to_save)
                self.save_variable(variable=cost_iters, name_of_variable="cost_iters_" + str(checkpoint_index), path_to_save=path_to_save)
                self.save_np_array_to_txt(variable=cost_iters, name_of_variable="cost_iters_"+str(checkpoint_index), path_to_save=path_to_save)
                self.save_variable(variable=iteration_index, name_of_variable="iteration_index_" + str(checkpoint_index), path_to_save=path_to_save)
                self.save_np_array_to_txt(variable=np.asarray(iteration_index), name_of_variable="iteration_index_"+str(checkpoint_index), path_to_save=path_to_save)
                self.save_variable(variable=eta_iters, name_of_variable="eta_iters_" + str(checkpoint_index), path_to_save=path_to_save)
                self.save_np_array_to_txt(variable=eta_iters, name_of_variable="eta_iters_"+str(checkpoint_index), path_to_save=path_to_save)

    def CPE_cost(self, X, Y):
        neighbor_indices = self.neighbor_indices
        n_samples = X.shape[1]
        costs = np.zeros((n_samples,))
        for sample_index in range(n_samples):
            sample_x = X[:, sample_index].reshape((-1, 1))
            sample_y = Y[:, sample_index].reshape((-1, 1))
            # ----- local cost:
            the_sum = 0
            for neighbor1_index in range(self.n_neighbors - 1):
                the_index = neighbor_indices[sample_index, neighbor1_index].reshape((-1, 1))
                the_index = int(the_index)
                x_a = X[:, the_index].reshape((-1, 1))
                x_a_hat = x_a - sample_x
                y_a = Y[:, the_index].reshape((-1, 1))
                y_a_hat = y_a - sample_y
                for neighbor2_index in range(neighbor1_index, self.n_neighbors):
                    the_index = neighbor_indices[sample_index, neighbor2_index].reshape((-1, 1))
                    the_index = int(the_index)
                    x_b = X[:, the_index].reshape((-1, 1))
                    x_b_hat = x_b - sample_x
                    y_b = Y[:, the_index].reshape((-1, 1))
                    y_b_hat = y_b - sample_y
                    cosine_x = self.cosine_of_vectors(x1=x_a_hat, x2=x_b_hat)
                    cosine_y = self.cosine_of_vectors(x1=y_a_hat, x2=y_b_hat)
                    the_sum = the_sum + ((cosine_x - cosine_y)**2)
            cost_local = the_sum
            # ----- global cost:
            the_sum2 = 0
            for sample_index2 in range(n_samples):
                sample_x2 = X[:, sample_index2].reshape((-1, 1))
                sample_y2 = Y[:, sample_index2].reshape((-1, 1))
                temp = ((sample_x.T).dot(sample_x2) - (sample_y.T).dot(sample_y2)) ** 2
                the_sum2 = the_sum2 + temp
            cost_global = the_sum2
            # ----- total cost:
            # costs[sample_index] = cost_local + cost_global
            costs[sample_index] = cost_global
        cost = costs.mean()
        return cost

    def CPE_cost_gradient(self, sample_index, sample_x, sample_y, X, Y):
        neighbor_indices = self.neighbor_indices
        # ----- local cost gradient:
        the_sum = 0
        for neighbor1_index in range(self.n_neighbors - 1):
            the_index = neighbor_indices[sample_index, neighbor1_index].reshape((-1, 1))
            the_index = int(the_index)
            x_a = X[:, the_index].reshape((-1, 1))
            x_a_hat = x_a - sample_x
            y_a = Y[:, the_index].reshape((-1, 1))
            y_a_hat = y_a - sample_y
            for neighbor2_index in range(neighbor1_index, self.n_neighbors):
                the_index = neighbor_indices[sample_index, neighbor2_index].reshape((-1, 1))
                the_index = int(the_index)
                x_b = X[:, the_index].reshape((-1, 1))
                x_b_hat = x_b - sample_x
                y_b = Y[:, the_index].reshape((-1, 1))
                y_b_hat = y_b - sample_y
                cosine_x = self.cosine_of_vectors(x1=x_a_hat, x2=x_b_hat)
                cosine_y = self.cosine_of_vectors(x1=y_a_hat, x2=y_b_hat)
                temp1 = cosine_x - cosine_y
                temp2 = LA.norm(y_a_hat, 2) * LA.norm(y_b_hat, 2)
                temp3 = 1.0 / temp2
                temp4 = -1 * (y_a_hat + y_b_hat)
                temp5 = (y_a_hat.T).dot(y_b_hat)
                temp6 = (1 / ((LA.norm(y_a_hat, 2)) ** 2)) * y_a_hat
                temp7 = (1 / ((LA.norm(y_b_hat, 2)) ** 2)) * y_b_hat
                temp8 = temp1 * temp3 * (temp4 + (temp5 * (temp6 + temp7)))
                the_sum = the_sum + (2 * temp8)
        gradient_local = -1 * the_sum
        # ----- global cost gradient:
        the_sum2 = 0
        for sample_index2 in range(self.n_neighbors):
            sample_x2 = X[:, sample_index2].reshape((-1, 1))
            sample_y2 = Y[:, sample_index2].reshape((-1, 1))
            temp9 = (sample_x.T).dot(sample_x2) - (sample_y.T).dot(sample_y2)
            temp10 = temp9 * sample_y2
            the_sum2 = the_sum2 + (2 * temp10)
        gradient_global = -1 * the_sum2
        # gradient = gradient_local + gradient_global
        gradient = gradient_global
        return gradient

    def kernel_CPE_cost(self, X, Y):
        neighbor_indices = self.neighbor_indices
        n_samples = X.shape[1]
        costs = np.zeros((n_samples,))
        for sample_index in range(n_samples):
            sample_x = X[:, sample_index].reshape((-1, 1))
            sample_y = Y[:, sample_index].reshape((-1, 1))
            the_indices = neighbor_indices[sample_index, :]
            the_indices = the_indices.astype(int)
            X_neighbors = X[:, the_indices]
            kernel_cosine_x_matrix = self.kernel_cosine_of_vectors(x=sample_x, X_neighbors=X_neighbors)
            the_sum = 0
            for neighbor1_index in range(self.n_neighbors - 1):
                the_index = neighbor_indices[sample_index, neighbor1_index].reshape((-1, 1))
                the_index = int(the_index)
                y_a = Y[:, the_index].reshape((-1, 1))
                y_a_hat = y_a - sample_y
                for neighbor2_index in range(neighbor1_index, self.n_neighbors):
                    kernel_cosine_x = kernel_cosine_x_matrix[neighbor1_index, neighbor2_index]
                    the_index = neighbor_indices[sample_index, neighbor2_index].reshape((-1, 1))
                    the_index = int(the_index)
                    y_b = Y[:, the_index].reshape((-1, 1))
                    y_b_hat = y_b - sample_y
                    cosine_y = self.cosine_of_vectors(x1=y_a_hat, x2=y_b_hat)
                    the_sum = the_sum + ((kernel_cosine_x - cosine_y) ** 2)
            costs[sample_index] = the_sum
        cost = costs.mean()
        return cost

    def kernel_CPE_cost_gradient(self, sample_index, sample_x, sample_y, X, Y):
        neighbor_indices = self.neighbor_indices
        the_indices = neighbor_indices[sample_index, :]
        the_indices = the_indices.astype(int)
        X_neighbors = X[:, the_indices]
        kernel_cosine_x_matrix = self.kernel_cosine_of_vectors(x=sample_x, X_neighbors=X_neighbors)
        the_sum = 0
        for neighbor1_index in range(self.n_neighbors - 1):
            the_index = neighbor_indices[sample_index, neighbor1_index].reshape((-1, 1))
            the_index = int(the_index)
            y_a = Y[:, the_index].reshape((-1, 1))
            y_a_hat = y_a - sample_y
            for neighbor2_index in range(neighbor1_index, self.n_neighbors):
                kernel_cosine_x = kernel_cosine_x_matrix[neighbor1_index, neighbor2_index]
                the_index = neighbor_indices[sample_index, neighbor2_index].reshape((-1, 1))
                the_index = int(the_index)
                y_b = Y[:, the_index].reshape((-1, 1))
                y_b_hat = y_b - sample_y
                cosine_y = self.cosine_of_vectors(x1=y_a_hat, x2=y_b_hat)
                temp1 = kernel_cosine_x - cosine_y
                temp2 = LA.norm(y_a_hat, 2) * LA.norm(y_b_hat, 2)
                temp3 = 1.0 / temp2
                temp4 = -1 * (y_a_hat + y_b_hat)
                temp5 = (y_a_hat.T).dot(y_b_hat)
                temp6 = (1 / ((LA.norm(y_a_hat, 2)) ** 2)) * y_a_hat
                temp7 = (1 / ((LA.norm(y_b_hat, 2)) ** 2)) * y_b_hat
                temp8 = temp1 * temp3 * (temp4 + (temp5 * (temp6 + temp7)))
                the_sum = the_sum + (2 * temp8)
        gradient = -1 * the_sum
        return gradient

    def kernel_cosine_of_vectors(self, x, X_neighbors):
        # x: column vector
        # X_neighbors: columns are samples, rows are features
        X_neighbors = X_neighbors - x
        kernel_over_neighbors = pairwise_kernels(X=X_neighbors.T, Y=X_neighbors.T, metric=self.kernel)
        kernel_cosine_matrix = self.normalize_the_kernel(kernel_matrix=kernel_over_neighbors)
        return kernel_cosine_matrix

    def cosine_of_vectors(self, x1, x2):
        # x1, x2: column vectors
        numerator = (x1.T).dot(x2)
        numerator = numerator.ravel()[0]
        denominator = LA.norm(x1, 2) * LA.norm(x2, 2)
        cosine = numerator / denominator
        if cosine > 1:  # because of computer imperfect calculations
            cosine = 1
        return cosine

    def normalize_the_kernel(self, kernel_matrix):
        diag_kernel = np.diag(kernel_matrix)
        k = (1 / np.sqrt(diag_kernel)).reshape((-1, 1))
        normalized_kernel_matrix = np.multiply(kernel_matrix, k.dot(k.T))
        return normalized_kernel_matrix

    def find_KNN_inInputSpace(self, X):
        self.neighbor_indices = np.zeros((self.n_samples, self.n_neighbors))
        # --- KNN:
        connectivity_matrix = KNN(X=X.T, n_neighbors=self.n_neighbors, mode='connectivity', include_self=False, n_jobs=-1)
        connectivity_matrix = connectivity_matrix.toarray()
        # --- store indices of neighbors:
        for sample_index in range(self.n_samples):
            self.neighbor_indices[sample_index, :] = np.argwhere(connectivity_matrix[sample_index, :] == 1).ravel()

    def find_KNN_inFeatureSpace(self, X):
        self.neighbor_indices = np.zeros((self.n_samples, self.n_neighbors))
        # --- kernel:
        self.the_kernel = pairwise_kernels(X=X.T, Y=X.T, metric=self.kernel)
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

    def save_variable(self, variable, name_of_variable, path_to_save='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.pckl'
        f = open(file_address, 'wb')
        pickle.dump(variable, f)
        f.close()

    def save_np_array_to_txt(self, variable, name_of_variable, path_to_save='./'):
        if type(variable) is list:
            variable = np.asarray(variable)
        # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.txt'
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
        with open(file_address, 'w') as f:
            f.write(np.array2string(variable, separator=', '))

    def load_variable(self, name_of_variable, path='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        file_address = path + name_of_variable + '.pckl'
        f = open(file_address, 'rb')
        variable = pickle.load(f)
        f.close()
        return variable