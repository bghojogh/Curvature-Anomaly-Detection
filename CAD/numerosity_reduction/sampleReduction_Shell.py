import numpy as np
from numpy import linalg as LA


# Sample Reduction with Shell Extraction:
# Paper: An efficient instance selection algorithm to reconstruct training set for support vector machine
class SR_Shell_Extraction:

    def __init__(self, X, Y):
        # X: rows are features and columns are samples
        # Y: rows are features and columns are samples
        self.X = X
        self.Y = Y
        self.n_dimensions = X.shape[0]
        self.n_samples = X.shape[1]
        self.n_classes = None
        self.lambda_parameter = 0.8  #--> should be in range [0.8, 1]
        self.delta_parameter = 0.1  #--> should be a small value
        self.kept_prototypes_indices = np.ones((1, self.n_samples))

    def prototype_selection(self, mode="classification", percentage_kept_data=50):
        if mode == "classification":
            X_separated_classes, original_index_in_whole_dataset = self.separate_samples_of_classes(X=self.X, y=self.Y)
            self.n_classes = len(X_separated_classes)
            for class_index in range(self.n_classes):
                n_samples_in_class = X_separated_classes[class_index].shape[1]
                kept_prototypes_indices_in_class = np.ones((1, n_samples_in_class))
                iteration_index = 0
                exit_the_iteration_of_class = False
                X_in_class = X_separated_classes[class_index]
                while True:
                    iteration_index = iteration_index + 1
                    temp = kept_prototypes_indices_in_class.astype(int)[0]
                    X_in_class_reduced = X_separated_classes[class_index][:, temp==1]
                    radius_of_Reduction_Sphere = self.calculate_radius_of_Reduction_Sphere(X_in_class_reduced, iteration_index)
                    centroid_of_class = np.mean(X_in_class_reduced, axis=1)
                    for sample_index_in_class in range(X_in_class.shape[1]):
                        if kept_prototypes_indices_in_class[:, sample_index_in_class] == 0:  #--> if already is removed, leave it
                            continue
                        sample_in_class = X_in_class[:, sample_index_in_class]
                        distance = LA.norm(sample_in_class - centroid_of_class)
                        if distance < radius_of_Reduction_Sphere:
                            kept_prototypes_indices_in_class[:, sample_index_in_class] = 0
                            temp = kept_prototypes_indices_in_class.astype(int)[0]
                            percentage_of_data_which_is_left = ( np.count_nonzero(np.asarray(temp)) / len(temp) ) * 100
                            # print(percentage_of_data_which_is_left)
                            if percentage_of_data_which_is_left < percentage_kept_data:
                                # should not reduce anymore:
                                exit_the_iteration_of_class = True
                                break #---> break for the class
                            else:
                                original_index = original_index_in_whole_dataset[class_index][sample_index_in_class]
                                self.kept_prototypes_indices[:, original_index] = 0
                        if (distance == 0) and (radius_of_Reduction_Sphere == 0):  #--> means all the samples of class have been removed
                            exit_the_iteration_of_class = True
                            break  # ---> break for the class
                    if exit_the_iteration_of_class == True:
                        break
        elif mode == "clustering" or mode == "regression":
                kept_prototypes_indices_in_overall = np.ones((1, self.n_samples))
                iteration_index = 0
                exit_the_iteration_of_class = False
                while True:
                    iteration_index = iteration_index + 1
                    temp = kept_prototypes_indices_in_overall.astype(int)[0]
                    X_reduced = self.X[:, temp==1]
                    radius_of_Reduction_Sphere = self.calculate_radius_of_Reduction_Sphere(X_reduced, iteration_index)
                    centroid_of_class = np.mean(X_reduced, axis=1)
                    for sample_index in range(self.X.shape[1]):
                        if kept_prototypes_indices_in_overall[:, sample_index] == 0:  #--> if already is removed, leave it
                            continue
                        sample_in_class = self.X[:, sample_index]
                        distance = LA.norm(sample_in_class - centroid_of_class)
                        if distance < radius_of_Reduction_Sphere:
                            kept_prototypes_indices_in_overall[:, sample_index] = 0
                            temp = kept_prototypes_indices_in_overall.astype(int)[0]
                            percentage_of_data_which_is_left = ( np.count_nonzero(np.asarray(temp)) / len(temp) ) * 100
                            # print(percentage_of_data_which_is_left)
                            if percentage_of_data_which_is_left < percentage_kept_data:
                                # should not reduce anymore:
                                exit_the_iteration_of_class = True
                                break #---> break for the class
                            else:
                                original_index = sample_index
                                self.kept_prototypes_indices[:, original_index] = 0
                        if (distance == 0) and (radius_of_Reduction_Sphere == 0):  #--> means all the samples of class have been removed
                            exit_the_iteration_of_class = True
                            break  # ---> break for the class
                    if exit_the_iteration_of_class == True:
                        break
        return self.kept_prototypes_indices

    def calculate_radius_of_Reduction_Sphere(self, X_in_class, iteration_index):
        phi = self.delta_parameter * iteration_index
        n_samples_in_class = X_in_class.shape[1]
        part1 = (self.lambda_parameter + phi) / n_samples_in_class
        centroid_of_class = np.mean(X_in_class, axis=1)
        sum_of_distances = 0
        for sample_index_in_class in range(X_in_class.shape[1]):
            sample_in_class = X_in_class[:, sample_index_in_class]
            distance = LA.norm(sample_in_class - centroid_of_class)
            sum_of_distances = sum_of_distances + distance
        radius = part1 * sum_of_distances
        return radius

    def separate_samples_of_classes(self, X, y):  # it does not change the order of the samples within every class
        # X --> rows: features, columns: samples
        # return X_separated_classes --> each element of list --> rows: samples, columns: features
        y = np.asarray(y)
        y = y.reshape((-1, 1)).ravel()
        labels_of_classes = sorted(set(y.ravel().tolist()))
        n_samples = X.shape[1]
        n_dimensions = X.shape[0]
        n_classes = len(labels_of_classes)
        X_separated_classes = [np.empty((n_dimensions, 0))] * n_classes
        original_index_in_whole_dataset = [[]] * n_classes
        for class_index in range(n_classes):
            for sample_index in range(self.n_samples):
                if y[sample_index] == labels_of_classes[class_index]:
                    X_separated_classes[class_index] = np.column_stack((X_separated_classes[class_index], X[:, sample_index].reshape((-1,1))))
                    original_index_in_whole_dataset[class_index].append(sample_index)
        return X_separated_classes, original_index_in_whole_dataset