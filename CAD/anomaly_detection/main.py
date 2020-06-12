from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from skimage.transform import resize
from matplotlib import offsetbox
import pandas as pd
import scipy.io
import csv
import scipy.misc
import os
import math
from sklearn.model_selection import train_test_split   #--> https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_moons, make_blobs, samples_generator
from sklearn.ensemble import IsolationForest  #--> https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest
from sklearn import manifold
from my_CAD import My_CAD
from my_CPE import My_CPE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import seaborn as sns
from scipy.io import loadmat
from add_distortion import Add_distortion
import time
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.svm import OneClassSVM  #--> https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM
from sklearn.neighbors import LocalOutlierFactor as LOF  #--> https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope  #--> https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html#sklearn.covariance.EllipticEnvelope
import h5py


def main():
    dataset = "one_blob"  #--> MNIST, Frey, two_moons, one_blob, two_blobs, two_different_blobs, Breast_cancer, Pima, Speech, Thyroid,
                        # Satellite, optdigits, letter, arrhythmia, ionosphere, http, shuttle, wine, annthyroid, smtp, musk, cardio, vowels, lympho
    visual_experiments_anomaly_detection = True
    visual_experiments_numerosity_reduction = False
    manifold_leaning_experiment = False
    generate_synthetic_datasets_again = False
    find_anomalyPath_again = False
    create_noisy_images_again = False
    denoise_image_again = False
    denoising_experiments = False
    split_in_cross_validation_again = False
    anomaly_detection_AUC_experiments = False
    method = "kernel_CAD"  #--> iso_forest, CAD, kernel_CAD, LLE, CPE, kernel_CPE, one_class_SVM, LOF, covariance_estimator
    kernel = "polynomial"  #--> ‘rbf’, laplacian, ‘sigmoid’, ‘linear’, ‘cosine’, ‘polynomial’ --> https://scikit-learn.org/stable/modules/metrics.html#metrics

    # manifold_visual_experiment(method=method, kernel=kernel, generate_synthetic_datasets_again=generate_synthetic_datasets_again)
    if dataset == "Frey" or dataset == "MNIST" or dataset == "Breast_cancer" or dataset == "Pima" or dataset == "Speech" or dataset == "Thyroid" or \
       dataset == "Satellite" or dataset == "optdigits" or dataset == "letter" or dataset == "arrhythmia" or dataset == "ionosphere" or dataset == "http" or \
            dataset == "shuttle" or dataset == "wine" or dataset == "annthyroid" or dataset == "smtp" or dataset == "musk" or dataset == "cardio" or \
            dataset == "vowels" or dataset == "lympho":
        X_train, Y_train, X_train_in_folds, X_test_in_folds, y_train_in_folds, y_test_in_folds = read_dataset(dataset_name=dataset, split_in_cross_validation_again=split_in_cross_validation_again)
        # plt.imshow(X_train[:,10].reshape(28,20), cmap="gray")
        # plt.show()
    if manifold_leaning_experiment:
        MNIST_visual_experiment(method=method, X_train=X_train, Y_train=Y_train, kernel=kernel)
        path_to_save = "./output/" + dataset + "/" + method + "/"
        KNN_classification(method=method, X_train=X_train, Y_train=Y_train, path_to_save=path_to_save, max_n_components=2, n_neighbors=10, kernel=kernel)
    if visual_experiments_anomaly_detection:
        anomaly_visual_experiment(dataset=dataset, method=method, anomalyDetection=True, kernel=kernel, generate_synthetic_datasets_again=generate_synthetic_datasets_again, find_anomalyPath_again=find_anomalyPath_again)
    if visual_experiments_numerosity_reduction:
        anomaly_visual_experiment(dataset=dataset, method=method, anomalyDetection=False, kernel=kernel, generate_synthetic_datasets_again=generate_synthetic_datasets_again, find_anomalyPath_again=find_anomalyPath_again)
    if denoising_experiments:
        if create_noisy_images_again:
            # image_index_to_become_noisy = 10
            image_index_to_become_noisy = 17
            image_original = X_train[:,image_index_to_become_noisy].reshape(28,20)
            dataset, MSE_of_images, distortion_type_of_images = create_noisy_dataset(image_original=image_original, n_images_per_distortion_type=10)
            # save dataset:
            save_variable(variable=MSE_of_images, name_of_variable="MSE_of_images", path_to_save="./datasets/Frey_dataset/distorted/")
            save_np_array_to_txt(variable=MSE_of_images, name_of_variable="MSE_of_images", path_to_save="./datasets/Frey_dataset/distorted/")
            save_variable(variable=distortion_type_of_images, name_of_variable="distortion_type_of_images", path_to_save="./datasets/Frey_dataset/distorted/")
            save_np_array_to_txt(variable=distortion_type_of_images, name_of_variable="distortion_type_of_images", path_to_save="./datasets/Frey_dataset/distorted/")
            image_height = image_original.shape[0]
            image_width = image_original.shape[1]
            n_samples = dataset.shape[1]
            for image_index in range(n_samples):
                sample = dataset[:, image_index].reshape((image_height, image_width))
                save_image(image_array=sample, path_without_file_name="./datasets/Frey_dataset/distorted/", file_name=str(image_index) + ".tif")
                scale = 5
                sample_scaled = scipy.misc.imresize(arr=sample, size=scale * 100)
                save_image(image_array=sample_scaled, path_without_file_name="./datasets/Frey_dataset/distorted_scaled/", file_name=str(image_index) + ".tif")
        if denoise_image_again:
            n_iterations = 4000
            image_index_to_become_noisy = 10
            noisy_image_indices = [7,17,27,37,47,57]
            for noisy_image_index in noisy_image_indices:
                address_image = "./Frey_dataset/distorted/" + str(noisy_image_index) + ".tif"
                noisy_image = load_image(address_image)
                # plt.imshow(noisy_image, cmap="gray")
                # plt.show()
                start = time.time()
                path_coordinates = denoise_image(image_index_to_become_noisy=image_index_to_become_noisy, noisy_image_reshaped=noisy_image.ravel(), dataset_reshaped=X_train[:, 0:100], kernel=kernel, n_iterations=n_iterations)
                end = time.time()
                time_of_denoising = end - start
                time_of_denoising = np.asarray(time_of_denoising)
                print(time_of_denoising)
                # plt.imshow(path_coordinates[-1, :, -1].reshape(28,20), cmap="gray")
                # plt.show()
                path_to_save = "./output/" + dataset + "/denoised/image" + str(noisy_image_index) + "/"
                path_coordinates_justAnomaly = path_coordinates[-1, :, :].reshape(28*20, n_iterations)
                save_variable(variable=path_coordinates_justAnomaly, name_of_variable="path_coordinates_justAnomaly", path_to_save=path_to_save)
                save_np_array_to_txt(variable=time_of_denoising, name_of_variable="time_of_denoising", path_to_save=path_to_save)
                # show_iterations = [1000-1, 2000-1, 3000-1, 4000-1, 5000-1, 6000-1, 7000-1, 8000-1, 9000-1, 10000-1]
                show_iterations = [1-1, 10-1, 100-1, 500-1, 1000-1, 1500-1, 2000-1, 2500-1, 3000-1, 3500-1, 4000-1]
                for iteration_to_show in show_iterations:
                    image_in_iteration = path_coordinates[-1, :, iteration_to_show].reshape(28,20)
                    scale = 5
                    sample_scaled = scipy.misc.imresize(arr=image_in_iteration, size=scale * 100)
                    save_image(image_array=sample_scaled, path_without_file_name=path_to_save, file_name="itr="+str(iteration_to_show) + ".tif")
        else:
            noisy_image_indices = [7, 17, 27, 37, 47, 57]
            for noisy_image_index in noisy_image_indices:
                path_to_save = "./output/" + dataset + "/denoised/image" + str(noisy_image_index) + "/"
                path_coordinates_justAnomaly = load_variable(name_of_variable="path_coordinates_justAnomaly", path=path_to_save)
                # if noisy_image_index == 7:
                #     the_range_to_display = range(100)
                the_range_to_display = range(100)
                for iteration_to_show in the_range_to_display:
                    image_in_iteration = path_coordinates_justAnomaly[:, iteration_to_show].reshape(28,20)
                    scale = 5
                    sample_scaled = scipy.misc.imresize(arr=image_in_iteration, size=scale * 100)
                    save_image(image_array=sample_scaled, path_without_file_name=path_to_save+"iters/", file_name="itr="+str(iteration_to_show) + ".tif")
    if anomaly_detection_AUC_experiments:
        anomaly_detection_AUC_experiment(anomaly_method=method, kernel=kernel, dataset=dataset, X_train_in_folds=X_train_in_folds, X_test_in_folds=X_test_in_folds, y_train_in_folds=y_train_in_folds, y_test_in_folds=y_test_in_folds)


def read_dataset(dataset_name, split_in_cross_validation_again):
    if dataset_name == "MNIST":
        path_dataset_save = "./datasets/MNIST_dataset/"
        file = open(path_dataset_save + 'X_train_picked.pckl', 'rb')
        X_train_picked = pickle.load(file);
        file.close()
        file = open(path_dataset_save + 'X_test_picked.pckl', 'rb')
        X_test_picked = pickle.load(file);
        file.close()
        file = open(path_dataset_save + 'y_train_picked.pckl', 'rb')
        y_train_picked = pickle.load(file);
        file.close()
        file = open(path_dataset_save + 'y_test_picked.pckl', 'rb')
        y_test_picked = pickle.load(file);
        file.close()
        X_train = X_train_picked
        X_test = X_test_picked
        Y_train = y_train_picked
        Y_test = y_test_picked
        X_train = X_train.T / 255
        X_test = X_test.T / 255
        image_height = 28
        image_width = 28
        X_train = X_train - X_train.mean(axis=1).reshape((-1, 1))
        X_train_in_folds, X_test_in_folds, y_train_in_folds, y_test_in_folds = None, None, None, None
    elif dataset_name == "Frey":
        path_dataset_save = "./datasets/Frey_dataset/"
        data = loadmat(path_dataset_save + "frey_rawface.mat")
        X_train = data['ff']
        Y_train = None
        image_height = 28
        image_width = 20
        X_train_in_folds, X_test_in_folds, y_train_in_folds, y_test_in_folds = None, None, None, None
    elif dataset_name == 'Breast_cancer':
        path_dataset = "./datasets/Breast_cancer/"
        data = pd.read_csv(path_dataset+"wdbc_data.txt", sep=",", header=None)  # read text file using pandas dataFrame: https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas
        labels_of_classes = ['M', 'B']
        X, y = read_BreastCancer_dataset(data=data, labels_of_classes=labels_of_classes)
        X = X.astype(np.float64)  # ---> otherwise MDS has error --> https://stackoverflow.com/questions/16990996/multidimensional-scaling-fitting-in-numpy-pandas-and-sklearn-valueerror
    elif dataset_name == "Pima":
        path_dataset = "./datasets/Pima/"
        data = loadmat(path_dataset + "pima.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "Speech":
        path_dataset = "./datasets/Speech/"
        data = loadmat(path_dataset + "speech.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "Thyroid":
        path_dataset = "./datasets/Thyroid/"
        data = loadmat(path_dataset + "thyroid.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "Satellite":
        path_dataset = "./datasets/Satellite/"
        data = loadmat(path_dataset + "satellite.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "optdigits":
        path_dataset = "./datasets/optdigits/"
        data = loadmat(path_dataset + "optdigits.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "letter":
        path_dataset = "./datasets/letter/"
        data = loadmat(path_dataset + "letter.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "arrhythmia":
        path_dataset = "./datasets/letter/"
        data = loadmat(path_dataset + "letter.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "ionosphere":
        path_dataset = "./datasets/ionosphere/"
        data = loadmat(path_dataset + "ionosphere.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "http":
        path_dataset = "./datasets/http/"
        with h5py.File(path_dataset+'http.mat', 'r') as f:
            a = list(f['X'])
            b = list(f['y'])
        dimension0 = a[0].reshape((-1, 1))
        dimension1 = a[1].reshape((-1, 1))
        dimension2 = a[2].reshape((-1, 1))
        X = np.column_stack((dimension0, dimension1))
        X = np.column_stack((X, dimension2))
        y = b[0]
        y = y.astype(int)
    elif dataset_name == "shuttle":
        path_dataset = "./datasets/ionosphere/"
        data = loadmat(path_dataset + "ionosphere.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "wine":
        path_dataset = "./datasets/wine/"
        data = loadmat(path_dataset + "wine.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
        X = X.astype(np.float64)
    elif dataset_name == "annthyroid":
        path_dataset = "./datasets/annthyroid/"
        data = loadmat(path_dataset + "annthyroid.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "smtp":
        path_dataset = "./datasets/smtp/"
        with h5py.File(path_dataset + 'smtp.mat', 'r') as f:
            a = list(f['X'])
            b = list(f['y'])
        dimension0 = a[0].reshape((-1, 1))
        dimension1 = a[1].reshape((-1, 1))
        dimension2 = a[2].reshape((-1, 1))
        X = np.column_stack((dimension0, dimension1))
        X = np.column_stack((X, dimension2))
        y = b[0]
        y = y.astype(int)
    elif dataset_name == "musk":
        path_dataset = "./datasets/musk/"
        data = loadmat(path_dataset + "musk.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "cardio":
        path_dataset = "./datasets/cardio/"
        data = loadmat(path_dataset + "cardio.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "vowels":
        path_dataset = "./datasets/vowels/"
        data = loadmat(path_dataset + "vowels.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    elif dataset_name == "lympho":
        path_dataset = "./datasets/lympho/"
        data = loadmat(path_dataset + "lympho.mat")
        X = data['X']
        y = data['y']
        y = y.ravel()
        y = y.astype(int)
    if dataset_name == 'Breast_cancer' or dataset_name == "Pima" or dataset_name == "Speech" or dataset_name == "Thyroid" or dataset_name == "Satellite" or \
        dataset_name == "optdigits" or dataset_name == "letter" or dataset_name == "arrhythmia" or dataset_name == "ionosphere" or dataset_name == "http" or \
            dataset_name == "shuttle" or dataset_name == "wine" or dataset_name == "annthyroid" or dataset_name == "smtp" or dataset_name == "musk" or \
            dataset_name == "cardio" or dataset_name == "vowels" or dataset_name == "lympho":
        # --- cross validation:
        path_to_save = path_dataset + "/CV/"
        number_of_folds = 10
        if split_in_cross_validation_again:
            train_indices_in_folds, test_indices_in_folds, X_train_in_folds, X_test_in_folds, y_train_in_folds, y_test_in_folds = cross_validation(X=X, y=y, n_splits=number_of_folds)
            save_variable(train_indices_in_folds, 'train_indices_in_folds', path_to_save=path_to_save)
            save_variable(test_indices_in_folds, 'test_indices_in_folds', path_to_save=path_to_save)
            save_variable(X_train_in_folds, 'X_train_in_folds', path_to_save=path_to_save)
            save_variable(X_test_in_folds, 'X_test_in_folds', path_to_save=path_to_save)
            save_variable(y_train_in_folds, 'y_train_in_folds', path_to_save=path_to_save)
            save_variable(y_test_in_folds, 'y_test_in_folds', path_to_save=path_to_save)
        else:
            file = open(path_to_save + 'train_indices_in_folds.pckl', 'rb')
            train_indices_in_folds = pickle.load(file)
            file.close()
            file = open(path_to_save + 'test_indices_in_folds.pckl', 'rb')
            test_indices_in_folds = pickle.load(file)
            file.close()
            file = open(path_to_save + 'X_train_in_folds.pckl', 'rb')
            X_train_in_folds = pickle.load(file)
            file.close()
            file = open(path_to_save + 'X_test_in_folds.pckl', 'rb')
            X_test_in_folds = pickle.load(file)
            file.close()
            file = open(path_to_save + 'y_train_in_folds.pckl', 'rb')
            y_train_in_folds = pickle.load(file)
            file.close()
            file = open(path_to_save + 'y_test_in_folds.pckl', 'rb')
            y_test_in_folds = pickle.load(file)
            file.close()
        X_train, Y_train = None, None
    return X_train, Y_train, X_train_in_folds, X_test_in_folds, y_train_in_folds, y_test_in_folds

def anomaly_detection_AUC_experiment(anomaly_method, kernel, dataset, X_train_in_folds, X_test_in_folds, y_train_in_folds, y_test_in_folds):
    rng = np.random.RandomState(42)
    n_folds = len(X_train_in_folds)
    auc_test_array = np.zeros((n_folds,))
    auc_train_array = np.zeros((n_folds,))
    time_of_algorithm_test = np.zeros((n_folds,))
    time_of_algorithm_train = np.zeros((n_folds,))
    for fold_index in range(n_folds):
        X_train = X_train_in_folds[fold_index].T
        X_test = X_test_in_folds[fold_index].T
        y_train = y_train_in_folds[fold_index]
        y_test = y_test_in_folds[fold_index]
        if dataset == "Breast_cancer":
            y_train[y_train == 0] = -1
            y_test[y_test == 0] = -1
        if dataset == "Pima" or dataset == "Speech" or dataset == "Thyroid" or dataset == "Satellite" or dataset == "optdigits" or \
                dataset == "letter" or dataset == "arrhythmia" or dataset == "ionosphere" or dataset == "http" or dataset == "shuttle" or dataset == "wine" or \
                dataset == "annthyroid" or dataset == "smtp" or dataset == "musk" or dataset == "cardio" or dataset == "vowels" or dataset == "lympho":
            y_train[y_train == 1] = -1
            y_test[y_test == 1] = -1
            y_train[y_train == 0] = 1
            y_test[y_test == 0] = 1
        if fold_index == 0:
            y = list(y_train)
            y.extend(y_test)
            y = np.asarray(y)
            # print(y)
            percentage_of_anomalies = sum(y == -1) / len(y)
            print("percentage of the anomalies = " + str(percentage_of_anomalies))
        if anomaly_method == "iso_forest":
            clf = IsolationForest(random_state=rng)
            start = time.time()
            clf.fit(X=X_train.T)
            scores_train = clf.decision_function(X=X_train.T)
            end = time.time()
            time_of_algorithm_train[fold_index] = end - start
            start = time.time()
            scores_test = clf.decision_function(X=X_test.T)
            end = time.time()
            time_of_algorithm_test[fold_index] = end - start
        elif anomaly_method == "one_class_SVM":
            clf = OneClassSVM(gamma='auto')
            start = time.time()
            clf.fit(X=X_train.T)
            scores_train = clf.decision_function(X=X_train.T)
            end = time.time()
            time_of_algorithm_train[fold_index] = end - start
            start = time.time()
            scores_test = clf.decision_function(X=X_test.T)
            end = time.time()
            time_of_algorithm_test[fold_index] = end - start
        elif anomaly_method == "LOF":
            n_neighbors = 10
            clf = LOF(n_neighbors=n_neighbors, contamination=0.1)
            start = time.time()
            clf.fit(X=X_train.T)
            scores_train = clf.negative_outlier_factor_
            end = time.time()
            time_of_algorithm_train[fold_index] = end - start
            clf = LOF(n_neighbors=n_neighbors, novelty=True, contamination=0.1)
            start = time.time()
            clf.fit(X=X_train.T)
            scores_test = clf.decision_function(X=X_test.T)
            end = time.time()
            time_of_algorithm_test[fold_index] = end - start
        elif anomaly_method == "covariance_estimator":
            clf = EllipticEnvelope(random_state=rng)
            start = time.time()
            clf.fit(X=X_train.T)
            scores_train = clf.decision_function(X=X_train.T)
            end = time.time()
            time_of_algorithm_train[fold_index] = end - start
            start = time.time()
            scores_test = clf.decision_function(X=X_test.T)
            end = time.time()
            time_of_algorithm_test[fold_index] = end - start
        elif anomaly_method == "CAD":
            n_neighbors = 10
            my_CAD = My_CAD(n_neighbors=n_neighbors, n_components=None, anomalyDetection=True, kernel=kernel)
            start = time.time()
            my_CAD.CAD_fit(X=X_train, display_scree_plot=False)
            y_train_pred, scores_train = my_CAD.CAD_predict(X=X_train)
            end = time.time()
            time_of_algorithm_train[fold_index] = end - start
            start = time.time()
            y_test_pred, scores_test = my_CAD.CAD_predict_outOfSample(X=X_test, exclude_anomalies_from_neighbors=False)
            end = time.time()
            time_of_algorithm_test[fold_index] = end - start
        elif anomaly_method == "kernel_CAD":
            n_neighbors = 10
            my_CAD = My_CAD(n_neighbors=n_neighbors, n_components=None, anomalyDetection=True, kernel=kernel)
            start = time.time()
            my_CAD.kernel_CAD_fit(X=X_train, display_scree_plot=False)
            y_train_pred, scores_train = my_CAD.CAD_predict(X=X_train)
            end = time.time()
            if kernel == "rbf" or kernel == "polynomial" or kernel == "laplacian":
                scores_train = -1 * scores_train
            time_of_algorithm_train[fold_index] = end - start
            start = time.time()
            y_test_pred, scores_test = my_CAD.kernel_CAD_predict_outOfSample(X=X_test, exclude_anomalies_from_neighbors=False)
            end = time.time()
            if kernel == "rbf" or kernel == "polynomial" or kernel == "laplacian":
                scores_test = -1 * scores_test
            time_of_algorithm_test[fold_index] = end - start
        # scores_test = -1 * scores_test  #--> to have: the more score, the less anomaly
        fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, scores_test, pos_label=1) #--> https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
        fpr_train, tpr_train, thresholds_train = metrics.roc_curve(y_train, scores_train, pos_label=1)
        # plt.plot(fpr_test, tpr_test)
        # plt.show()
        # plt.plot(fpr_train, tpr_train)
        # plt.show()
        auc_test = metrics.auc(fpr_test, tpr_test)  #--> https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
        print("Fold: " + str(fold_index) + " ---> AUC for test: " + str(auc_test))
        auc_test_array[fold_index] = auc_test
        auc_train = metrics.auc(fpr_train, tpr_train)
        print("Fold: " + str(fold_index) + " ---> AUC for train: " + str(auc_train))
        auc_train_array[fold_index] = auc_train
    auc_test_mean = auc_test_array.mean()
    auc_test_std = auc_test_array.std()
    auc_train_mean = auc_train_array.mean()
    auc_train_std = auc_train_array.std()
    time_of_algorithm_train_mean = time_of_algorithm_train.mean()
    time_of_algorithm_train_std = time_of_algorithm_train.std()
    time_of_algorithm_test_mean = time_of_algorithm_test.mean()
    time_of_algorithm_test_std = time_of_algorithm_test.std()
    print("Average AUC for test data: " + str(auc_test_mean) + " +- " + str(auc_test_std))
    print("Average time for test data: " + str(time_of_algorithm_test_mean) + " +- " + str(time_of_algorithm_test_std))
    print("Average AUC for train data: " + str(auc_train_mean) + " +- " + str(auc_train_std))
    print("Average time for train data: " + str(time_of_algorithm_train_mean) + " +- " + str(time_of_algorithm_train_std))
    if anomaly_method == "LOF" or anomaly_method == "CAD":
        path = './output/' + dataset + "/" + anomaly_method + "/neigh=" + str(n_neighbors) + "/"
    elif anomaly_method == "kernel_CAD":
        path = './output/' + dataset + "/" + anomaly_method + "/" + kernel + "/neigh=" + str(n_neighbors) + "/"
    else:
        path = './output/' + dataset + "/" + anomaly_method + "/"
    save_np_array_to_txt(variable=auc_test_array, name_of_variable="auc_test_array", path_to_save=path)
    save_np_array_to_txt(variable=auc_test_mean, name_of_variable="auc_test_mean", path_to_save=path)
    save_np_array_to_txt(variable=auc_test_std, name_of_variable="auc_test_std", path_to_save=path)
    save_np_array_to_txt(variable=auc_train_array, name_of_variable="auc_train_array", path_to_save=path)
    save_np_array_to_txt(variable=auc_train_mean, name_of_variable="auc_train_mean", path_to_save=path)
    save_np_array_to_txt(variable=auc_train_std, name_of_variable="auc_train_std", path_to_save=path)
    save_np_array_to_txt(variable=time_of_algorithm_test, name_of_variable="time_of_algorithm_test", path_to_save=path)
    save_np_array_to_txt(variable=time_of_algorithm_test_mean, name_of_variable="time_of_algorithm_test_mean", path_to_save=path)
    save_np_array_to_txt(variable=time_of_algorithm_test_std, name_of_variable="time_of_algorithm_test_std", path_to_save=path)
    save_np_array_to_txt(variable=time_of_algorithm_train, name_of_variable="time_of_algorithm_train", path_to_save=path)
    save_np_array_to_txt(variable=time_of_algorithm_train_mean, name_of_variable="time_of_algorithm_train_mean", path_to_save=path)
    save_np_array_to_txt(variable=time_of_algorithm_train_std, name_of_variable="time_of_algorithm_train_std", path_to_save=path)
    save_np_array_to_txt(variable=percentage_of_anomalies, name_of_variable="percentage_of_anomalies", path_to_save=path)


def denoise_image(image_index_to_become_noisy, noisy_image_reshaped, dataset_reshaped, kernel, n_iterations):
    version = 2
    my_CAD = My_CAD(n_neighbors=3, n_components=None, anomalyDetection=True, kernel=kernel)
    # n_images_in_dataset = dataset_reshaped.shape[1]
    dataset_reshaped = np.column_stack((dataset_reshaped[:, :image_index_to_become_noisy], dataset_reshaped[:, (image_index_to_become_noisy+1):]))
    n_images_in_dataset = dataset_reshaped.shape[1]
    X = np.column_stack((dataset_reshaped, noisy_image_reshaped))
    forced_y_pred = [1] * n_images_in_dataset
    forced_y_pred.extend([-1])
    forced_y_pred = np.asarray(forced_y_pred)
    my_CAD.set_y_pred(forced_y_pred=forced_y_pred, X=X)
    path_coordinates, y_pred = my_CAD.find_anomaly_paths(n_iterations=n_iterations, version=version, eta = 10000, termination_check=False)
    return path_coordinates

def create_noisy_dataset(image_original, n_images_per_distortion_type=10):
    image_height = image_original.shape[0]
    image_width = image_original.shape[1]
    # create dataset of noisy images:
    distortion_types = ["contrast_strech", "gaussian_noise", "mean_shift", "gaussian_blurring", "salt_and_pepper", "jpeg_distortion"]
    n_features = image_height * image_width
    n_samples = 1 + (n_images_per_distortion_type * len(distortion_types))  # +1 because the first image is the original image
    dataset = np.zeros(shape=(n_features, n_samples))
    MSE_of_images = np.zeros(shape=(1, n_samples))
    distortion_type_of_images = []
    dataset[:, 0] = image_original.ravel()  # original image as the first image
    MSE_of_images[:, 0] = 0  # original image as the first image
    distortion_type_of_images.append("original")  # original image as the first image
    add_distortion = Add_distortion(image_original=image_original)
    max_MSE = 900
    for distortion_type_index, distortion_type in enumerate(distortion_types):
        print("==== Distortion type: " + distortion_type)
        for image_index in range(1, n_images_per_distortion_type+1):
            print("---- image index: " + str(image_index) + " out of " + str(n_images_per_distortion_type) + " images")
            desired_MSE = (max_MSE / n_images_per_distortion_type) * (image_index)
            if distortion_type == "contrast_strech":
                initial_distortion = 255/4
            elif distortion_type == "gaussian_noise":
                initial_distortion = 10
            elif distortion_type == "mean_shift":
                initial_distortion = 0
            elif distortion_type == "gaussian_blurring":
                initial_distortion = 1
            elif distortion_type == "salt_and_pepper":
                initial_distortion = 0.5
            elif distortion_type == "jpeg_distortion":
                initial_distortion = 10
            distorted_image, MSE = add_distortion.add_distrotion_for_an_MSE_level(desired_MSE=desired_MSE, distrotion_type=distortion_type, initial_distortion=initial_distortion)
            print("------------ image created with MSE = " + str(MSE) + " | desired MSE was " + str(desired_MSE))
            dataset[:, (distortion_type_index*n_images_per_distortion_type)+image_index] = distorted_image.ravel()
            MSE_of_images[:, (distortion_type_index*n_images_per_distortion_type)+image_index] = MSE
            distortion_type_of_images.append(distortion_type)
    return dataset, MSE_of_images, distortion_type_of_images

def MNIST_visual_experiment(method, X_train, Y_train, kernel):
    # --- manifold embedding:
    if method == "LLE":
        X_embedded, err = manifold.locally_linear_embedding(X_train.T, n_neighbors=12, n_components=2)
        X_embedded = X_embedded.T
        # pca = PCA(n_components=2)
        # X_embedded = pca.fit_transform(X=X_train.T)
        # X_embedded = X_embedded.T
        # mds = MDS(n_components=2)
        # X_embedded = mds.fit_transform(X=X_train.T)
        # X_embedded = X_embedded.T
    elif method == "CPE":
        cpe = My_CPE(n_neighbors=10, n_components=2, kernel=kernel)
        X_embedded = cpe.CPE_fit(X=X_train, step_checkpoint=5, fit_again=False)
    elif method == "kernel_CPE":
        cpe = My_CPE(n_neighbors=10, n_components=2, kernel=kernel)
        X_embedded = cpe.kernel_CPE_fit(X=X_train, step_checkpoint=5, fit_again=False)
    scatter_plot_MNIST(data_transformed=X_embedded, which_dimensions_to_plot=[0,1], labels=Y_train.ravel(), data_test_transformed=None, show_projected_test=False)

def manifold_visual_experiment(method, kernel, generate_synthetic_datasets_again):
    # https://scikit-learn.org/stable/auto_examples/manifold/plot_swissroll.html
    if generate_synthetic_datasets_again:
        X, color = samples_generator.make_swiss_roll(n_samples=1500)
        save_variable(variable=X, name_of_variable="X", path_to_save='./datasets/Swiss_roll/')
        save_variable(variable=color, name_of_variable="color", path_to_save='./datasets/Swiss_roll/')
    else:
        X = load_variable(name_of_variable="X", path='./datasets/Swiss_roll/')
        color = load_variable(name_of_variable="color", path='./datasets/Swiss_roll/')
    if method == "LLE":
        X_embedded, err = manifold.locally_linear_embedding(X, n_neighbors=12, n_components=2)
    elif method == "CPE":
        cpe = My_CPE(n_neighbors=10, n_components=2, kernel=kernel)
        X_embedded = cpe.CPE_fit(X=X.T, step_checkpoint=5, fit_again=False)
        X_embedded = X_embedded.T
    elif method == "kernel_CPE":
        cpe = My_CPE(n_neighbors=10, n_components=2, kernel=kernel)
        X_embedded = cpe.kernel_CPE_fit(X=X.T, step_checkpoint=5, fit_again=False)
        X_embedded = X_embedded.T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])
    plt.show()

def anomaly_visual_experiment(dataset, method, anomalyDetection, kernel, generate_synthetic_datasets_again, find_anomalyPath_again):
    # https://scikit-learn.org/stable/auto_examples/plot_anomaly_comparison.html#sphx-glr-auto-examples-plot-anomaly-comparison-py
    # settings:
    n_samples = 300
    outliers_fraction = 0.15
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers
    xx, yy = np.meshgrid(np.linspace(-7, 7, 150), np.linspace(-7, 7, 150))
    rng = np.random.RandomState(42)
    # dataset:
    path_dataset = './datasets/' + dataset + "/"
    if generate_synthetic_datasets_again:
        if dataset == "two_moons":
            X = 4. * (make_moons(n_samples=n_inliers, noise=.05, random_state=0)[0] - np.array([0.5, 0.25]))
        elif dataset == "one_blob":
            X = make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5)[0]
        elif dataset == "two_blobs":
            X = make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5])[0]
        elif dataset == "two_different_blobs":
            X = make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3])[0]
        save_variable(variable=X, name_of_variable="X", path_to_save=path_dataset)
    else:
        X = load_variable(name_of_variable="X", path=path_dataset)
    # Add outliers:
    X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)
    # transpose (to have columns as features and rows as samples):
    X = X.T
    # anomaly detection algorithm:
    if method == "iso_forest":
        iso_forest = IsolationForest(contamination=outliers_fraction, random_state=42)
        algorithm = iso_forest
        algorithm.fit(X.T)
        y_pred = algorithm.predict(X.T)
        print(y_pred)
        Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    elif method == "CAD":
        my_CAD = My_CAD(n_neighbors=10, n_components=None, anomalyDetection=anomalyDetection, kernel=kernel)
        algorithm = my_CAD
        algorithm.CAD_fit(X=X, display_scree_plot=False)
        y_pred, scores = algorithm.CAD_predict(X=X)
        print(y_pred)
        if not anomalyDetection:
            ranks = algorithm.CAD_rank_instances(scores, kernel_approach=False)
            X_sorted, Y_sorted, scores_sorted = algorithm.sort_samples(scores=scores, X=X, Y=None, kernel_approach=False)
        if anomalyDetection:
            predict_out_of_sample_again = False
            path_save = "./CAD_settings/" + dataset + "/" + method + "/outOfSample/"
            if predict_out_of_sample_again:
                Z, scores = algorithm.CAD_predict_outOfSample(X=np.c_[xx.ravel(), yy.ravel()].T, exclude_anomalies_from_neighbors=False)
                _, scores_for_anomalyPaths = algorithm.CAD_predict_outOfSample(X=np.c_[xx.ravel(), yy.ravel()].T, exclude_anomalies_from_neighbors=True)
                save_variable(variable=Z, name_of_variable="Z", path_to_save=path_save)
                save_variable(variable=scores, name_of_variable="scores", path_to_save=path_save)
                save_variable(variable=scores_for_anomalyPaths, name_of_variable="scores_for_anomalyPaths", path_to_save=path_save)
            else:
                Z = load_variable(name_of_variable="Z", path=path_save)
                scores = load_variable(name_of_variable="scores", path=path_save)
                scores_for_anomalyPaths = load_variable(name_of_variable="scores_for_anomalyPaths", path=path_save)
            Z = Z.reshape(xx.shape)
            # print(scores.shape)
            scores = scores.reshape(xx.shape)
            scores_for_anomalyPaths = scores_for_anomalyPaths.reshape(xx.shape)
            # print(scores.shape)
            # print(list(scores))
    elif method == "kernel_CAD":
        my_CAD = My_CAD(n_neighbors=10, n_components=None, anomalyDetection=anomalyDetection, kernel=kernel)
        algorithm = my_CAD
        algorithm.kernel_CAD_fit(X=X, display_scree_plot=False)
        y_pred, scores = algorithm.CAD_predict(X=X)
        print(y_pred)
        if not anomalyDetection:
            ranks = algorithm.CAD_rank_instances(scores, kernel_approach=True)
            X_sorted, Y_sorted, scores_sorted = algorithm.sort_samples(scores=scores, X=X, Y=None, kernel_approach=True)
        if anomalyDetection:
            predict_out_of_sample_again = True
            path_save = "./CAD_settings/" + dataset + "/" + method + "/" + kernel + "/outOfSample/"
            if predict_out_of_sample_again:
                Z, scores = algorithm.kernel_CAD_predict_outOfSample(X=np.c_[xx.ravel(), yy.ravel()].T, exclude_anomalies_from_neighbors=False)
                _, scores_for_anomalyPaths = algorithm.CAD_predict_outOfSample(X=np.c_[xx.ravel(), yy.ravel()].T, exclude_anomalies_from_neighbors=True)
                save_variable(variable=Z, name_of_variable="Z", path_to_save=path_save)
                save_variable(variable=scores, name_of_variable="scores", path_to_save=path_save)
                # save_variable(variable=scores_for_anomalyPaths, name_of_variable="scores_for_anomalyPaths", path_to_save=path_save)
            else:
                Z = load_variable(name_of_variable="Z", path=path_save)
                scores = load_variable(name_of_variable="scores", path=path_save)
                # scores_for_anomalyPaths = load_variable(name_of_variable="scores_for_anomalyPaths", path=path_save)
            Z = Z.reshape(xx.shape)
            scores = scores.reshape(xx.shape)
            # scores_for_anomalyPaths = scores_for_anomalyPaths.reshape(xx.shape)
    if anomalyDetection:
        # ------ legends:
        colors = np.array(['#377eb8', '#ff7f00'])
        markers = np.array(['^', 'o'])
        plt.scatter(0, 0, color=colors[1], marker=markers[1])
        plt.scatter(1, 1, color=colors[0], marker=markers[0])
        plt.legend(["normal", "anomaly"])
        plt.show()
        # ------ plot the predicted anomaly for the space:
        # plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
        # plt.imshow(scores, cmap='hot', interpolation='nearest')
        plt.imshow(Z * -1, cmap='gray', alpha=0.2)
        # plt.colorbar()
        colors = np.array(['#377eb8', '#ff7f00'])
        markers = np.array(['^', 'o'])
        # plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2], marker="o")
        colors_vector = colors[(y_pred + 1) // 2]
        markers_vector = markers[(y_pred + 1) // 2]
        for _s, c, _x, _y in zip(markers_vector, colors_vector, X[0, :], X[1, :]):
            _x = (_x + 7) * (150 / 14)
            _y = (_y + 7) * (150 / 14)
            plt.scatter(_x, _y, marker=_s, c=c, alpha=1)
        plt.xlim(0, 150)
        plt.ylim(0, 150)
        # plt.xlim(-7, 7)
        # plt.ylim(-7, 7)
        plt.xticks(())
        plt.yticks(())
        plt.show()
        # ------ plot the anomaly score for the space:
        plt.imshow(scores, cmap='gray')
        plt.colorbar()
        # plt.xlim(-7, 7)
        # plt.ylim(-7, 7)
        # plt.show()
        colors = np.array(['#377eb8', '#ff7f00'])
        markers = np.array(['^', 'o'])
        # plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2], marker="o")
        colors_vector = colors[(y_pred + 1) // 2]
        markers_vector = markers[(y_pred + 1) // 2]
        for _s, c, _x, _y in zip(markers_vector, colors_vector, X[0, :], X[1, :]):
            _x = (_x + 7) * (150 / 14)
            _y = (_y + 7) * (150 / 14)
            plt.scatter(_x, _y, marker=_s, c=c, alpha=0.25)
        plt.xlim(0, 150)
        plt.ylim(0, 150)
        # plt.xlim(-7, 7)
        # plt.ylim(-7, 7)
        plt.xticks(())
        plt.yticks(())
        plt.show()
        # ------ plot the anomaly paths:
        if method == "CAD":
            version = 2
            path_save = "./CAD_settings/" + dataset + "/CAD/anomaly_path/version" + str(version) + "/"
            if find_anomalyPath_again:
                path_coordinates, y_pred = my_CAD.find_anomaly_paths(n_iterations=10000, version=version)
                save_variable(variable=path_coordinates, name_of_variable="path_coordinates", path_to_save=path_save)
                save_variable(variable=y_pred, name_of_variable="y_pred", path_to_save=path_save)
            else:
                path_coordinates = load_variable(name_of_variable="path_coordinates", path=path_save)
                y_pred = load_variable(name_of_variable="y_pred", path=path_save)
            plt.imshow(scores_for_anomalyPaths, cmap='gray')
            plt.colorbar()
            for _s, c, _x, _y in zip(markers_vector, colors_vector, X[0, :], X[1, :]):
                _x = (_x + 7) * (150 / 14)
                _y = (_y + 7) * (150 / 14)
                plt.scatter(_x, _y, marker=_s, c=c, alpha=0.25)
            plt.xlim(0, 150)
            plt.ylim(0, 150)
            # plt.xlim(-7, 7)
            # plt.ylim(-7, 7)
            plt.xticks(())
            plt.yticks(())
            n_samples = X.shape[1]
            for sample_index in range(n_samples):
                if y_pred[sample_index] == -1:
                    # plt.plot(path_coordinates[sample_index, 0, :], path_coordinates[sample_index, 1, :], 'r-')
                    x = path_coordinates[sample_index, 0, :]
                    x = (x + 7) * (150 / 14)
                    y = path_coordinates[sample_index, 1, :]
                    y = (y + 7) * (150 / 14)
                    plt.plot(x, y, 'r-')
                    # input("hiiiii")
            plt.show()
    else:
        # plot_final_ranked_samples(X, ranks=ranks)
        plot_final_ranked_samples_2(X_sorted=X_sorted)

def plot_final_ranked_samples(X, ranks):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plt1 = []
    number_of_samples = X.shape[1]
    for sample_index in range(number_of_samples):
        # rank_among_all_samples = ranks[sample_index]
        rank_among_all_samples = int(np.where(ranks == sample_index)[0])  # find place of sample_index in Individual_ranks[class_index][:]
        # print(rank_among_all_samples)
        biggest_marker_size = 100; smallest_marker_size = 1;
        step_marker_size = (biggest_marker_size-smallest_marker_size)/number_of_samples
        marker_size=biggest_marker_size-(step_marker_size*rank_among_all_samples)
        # if marker_size < (biggest_marker_size-smallest_marker_size)/1.25:
        #     marker_size = np.log(marker_size)
        if rank_among_all_samples == number_of_samples-10:   # a small marker
            plt1 = plt.scatter(X[0,sample_index], X[1,sample_index], c='b', marker='o', s=marker_size, edgecolors='k')
        else:
            plt.scatter(X[0,sample_index], X[1,sample_index], c='b', marker='o', s=marker_size, edgecolors='k')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    plt.xticks(())
    plt.yticks(())
    plt.show()

def plot_final_ranked_samples_2(X_sorted):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plt1 = []
    number_of_samples = X_sorted.shape[1]
    for sample_index in range(number_of_samples):
        x = X_sorted[:, sample_index]
        rank_among_all_samples = sample_index
        biggest_marker_size = 100; smallest_marker_size = 1
        step_marker_size = (biggest_marker_size-smallest_marker_size)/number_of_samples
        marker_size=biggest_marker_size-(step_marker_size*rank_among_all_samples)
        # if marker_size < (biggest_marker_size-smallest_marker_size)/1.25:
        #     marker_size = np.log(marker_size)
        if rank_among_all_samples == number_of_samples-10:   # a small marker
            plt1 = plt.scatter(x[0], x[1], c='b', marker='o', s=marker_size, edgecolors='k')
        else:
            plt.scatter(x[0], x[1], c='b', marker='o', s=marker_size, edgecolors='k')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    plt.xticks(())
    plt.yticks(())
    plt.show()

def scatter_plot_MNIST(data_transformed, which_dimensions_to_plot, labels, data_test_transformed=None, show_projected_test=False):
    colors_of_labels = ["green", "blue", "black", "fuchsia", "y", "orange", "red", "slategrey", "slateblue", "olive"]
    used_labels = []
    for sample_index in range(data_transformed.shape[1]):
        # list of colors --> https: // matplotlib.org / examples / color / named_colors.html
        label_numeric = (labels[sample_index]).astype(int)
        # print(label_numeric)
        color = colors_of_labels[label_numeric]
        label = str(label_numeric)
        if label in used_labels:
            label = ""
        else:
            used_labels.append(label)
        plt.scatter(data_transformed[which_dimensions_to_plot[0], sample_index], data_transformed[which_dimensions_to_plot[1], sample_index], label=label, color=color, marker="o", s=50)
    plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
    plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
    plt.legend()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    g = ( max(data_transformed[which_dimensions_to_plot[0], :]) - min(data_transformed[which_dimensions_to_plot[0], :]) ) * 0.1
    plt.xlim([min(data_transformed[which_dimensions_to_plot[0], :]) - g, max(data_transformed[which_dimensions_to_plot[0], :]) + g])
    g = (max(data_transformed[which_dimensions_to_plot[1], :]) - min(data_transformed[which_dimensions_to_plot[1], :])) * 0.1
    plt.ylim([min(data_transformed[which_dimensions_to_plot[1], :]) - g, max(data_transformed[which_dimensions_to_plot[1], :]) + g])
    plt.xticks([])
    plt.yticks([])
    if show_projected_test:
        min_x_axis = min(data_transformed[which_dimensions_to_plot[0], :])
        max_x_axis = max(data_transformed[which_dimensions_to_plot[0], :])
        min_y_axis = min(data_transformed[which_dimensions_to_plot[1], :])
        max_y_axis = max(data_transformed[which_dimensions_to_plot[1], :])
        for test_sample_index in range(data_test_transformed.shape[1]):
            dim1 = data_test_transformed[which_dimensions_to_plot[0], test_sample_index]
            dim2 = data_test_transformed[which_dimensions_to_plot[1], test_sample_index]
            plt.scatter(dim1, dim2, label=label, color="magenta", marker="s", s=80, alpha=1)
            plt.annotate(test_sample_index+1, (dim1, dim2)) #https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
            min_x_axis = min(dim1, min_x_axis)
            max_x_axis = max(dim1, max_x_axis)
            min_y_axis = min(dim2, min_y_axis)
            max_y_axis = max(dim2, max_y_axis)
        plt.xlim([min_x_axis, max_x_axis])
        plt.ylim([min_y_axis, max_y_axis])
    plt.show()

# def plot_error_MNIST_in_RDA(dataset="RDA", show_legends=True):
#     fig, ax = plt.subplots(figsize=(10, 10))
#     x = np.arange(0, 63+1, 1)
#     path = './values/' + dataset + '/'
#     levels = load_variable(name_of_variable="best_number_of_levels", path=path)
#     plt.bar(x=x, height=levels)
#     plt.grid()
#     plt.xlabel("Frequency", fontsize=13)
#     plt.ylabel("Optimum # quantization levels", fontsize=13)
#     # if show_legends is not None:
#     #     ax.legend()
#     plt.show()

def KNN_classification(method, X_train, Y_train, path_to_save, kernel, max_n_components=20, n_neighbors=1):
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    if method == "LLE":
        X_embedded, err = manifold.locally_linear_embedding(X_train.T, n_neighbors=12, n_components=2)
        X_embedded = X_embedded.T
    elif method == "CPE":
        cpe = My_CPE(n_neighbors=10, n_components=2, kernel=kernel)
        X_embedded = cpe.CPE_fit(X=X_train, step_checkpoint=5, fit_again=False)
    elif method == "kernel_CPE":
        cpe = My_CPE(n_neighbors=10, n_components=2, kernel=kernel)
        X_embedded = cpe.kernel_CPE_fit(X=X_train, step_checkpoint=5, fit_again=False)
    error_train = np.zeros((max_n_components,))
    for n_component in range(1,max_n_components+1):
        X_embedded_truncated = X_embedded[:n_component, :]
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
        neigh.fit(X=X_embedded_truncated.T, y=Y_train.ravel())
        predicted_Y_train = neigh.predict(X=X_embedded_truncated.T)
        error_train[n_component-1] = sum(predicted_Y_train != Y_train) / len(Y_train)
        print("#components: "+str(n_component)+", error: "+str(error_train[n_component-1]))
    # train error:
    error_train_mean = error_train.mean()
    error_train_std = error_train.std()
    error_train_result = np.array([error_train_mean, error_train_std])
    # save:
    save_np_array_to_txt(variable=error_train, name_of_variable="error_train", path_to_save=path_to_save)
    save_np_array_to_txt(variable=error_train_result, name_of_variable="error_train_result", path_to_save=path_to_save)

def save_variable(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.pckl'
    f = open(file_address, 'wb')
    pickle.dump(variable, f)
    f.close()

def load_variable(name_of_variable, path='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    file_address = path + name_of_variable + '.pckl'
    f = open(file_address, 'rb')
    variable = pickle.load(f)
    f.close()
    return variable

def save_np_array_to_txt(variable, name_of_variable, path_to_save='./'):
    if type(variable) is list:
        variable = np.asarray(variable)
    # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.txt'
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    with open(file_address, 'w') as f:
        f.write(np.array2string(variable, separator=', '))

def save_image(image_array, path_without_file_name, file_name):
    if not os.path.exists(path_without_file_name):
        os.makedirs(path_without_file_name)
    # http://code.activestate.com/recipes/577591-conversion-of-pil-image-and-numpy-array/
    img = Image.fromarray(image_array)
    img = img.convert("L")
    img.save(path_without_file_name + file_name)

def load_image(address_image):
    # http://code.activestate.com/recipes/577591-conversion-of-pil-image-and-numpy-array/
    img = Image.open(address_image).convert('L')
    img_arr = np.array(img)
    return img_arr

def read_BreastCancer_dataset(data, labels_of_classes):
    data = data.values  # converting pandas dataFrame to numpy array
    labels = data[:,1]
    total_number_of_samples = data.shape[0]
    X = data[:,2:]
    X = X.astype(np.float32)  # if we don't do that, we will have this error: https://www.reddit.com/r/learnpython/comments/7ivopz/numpy_getting_error_on_matrix_inverse/
    y = [None] * (total_number_of_samples)  # numeric labels
    for sample_index in range(total_number_of_samples):
        if labels[sample_index] == labels_of_classes[0]:  # first class --> M
            y[sample_index] = 0
        elif labels[sample_index] == labels_of_classes[1]:  # second class --> B
            y[sample_index] = 1
    return X, y

def cross_validation(X, y, n_splits=10):
    # sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=None)
    CV = KFold(n_splits=n_splits, random_state=100, shuffle=True)
    train_indices_in_folds = []; test_indices_in_folds = []
    X_train_in_folds = []; X_test_in_folds = []
    y_train_in_folds = []; y_test_in_folds = []
    for train_index, test_index in CV.split(X, y):
        train_indices_in_folds.append(train_index)
        test_indices_in_folds.append(test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np.asarray(y)[train_index], np.asarray(y)[test_index]
        X_train_in_folds.append(X_train)
        X_test_in_folds.append(X_test)
        y_train_in_folds.append(y_train)
        y_test_in_folds.append(y_test)
    return train_indices_in_folds, test_indices_in_folds, X_train_in_folds, X_test_in_folds, y_train_in_folds, y_test_in_folds

if __name__ == '__main__':
    main()