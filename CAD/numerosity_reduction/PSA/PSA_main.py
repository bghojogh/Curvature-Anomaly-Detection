import pickle
import random
from PSA.GroupRanking_class import *
from PSA.IndividualRankingInGroups_class import *
from PSA.IndividualRankingInUnqualifieds_class import *
from PSA.Plot_tools import *
import os
import numpy as np
from sklearn import decomposition


class PSA:

    def __init__(self, save_ranks=True, add_noisy_samples_if_not_necessary=False, number_of_added_noisy_samples=0,
                 number_of_selected_samples_in_group=10, number_of_iterations_of_RANSAC=20):
        # Parameters:
        self.add_noisy_samples_if_not_necessary = add_noisy_samples_if_not_necessary
        self.number_of_added_noisy_samples = number_of_added_noisy_samples
        self.number_of_selected_samples_in_group = number_of_selected_samples_in_group
        self.number_of_iterations_of_RANSAC = number_of_iterations_of_RANSAC
        self.save_ranks = save_ranks
        # these variables are used only for demo:
        self.number_of_classes = 3
        self.dimension_of_data = 2
        self.number_of_samples_of_each_class = [50, 50, 50]

    def set_demo_settings(self, number_of_classes, dimension_of_data, number_of_samples_of_each_class):
        self.number_of_classes = number_of_classes
        self.dimension_of_data = dimension_of_data
        self.number_of_samples_of_each_class = number_of_samples_of_each_class

    def rank_samples(self, X, y, demo_mode=False, report_steps=False, HavingClasses=True, Apply_PCA=False, n_components=None, for_regression_task=False):
        """
        :param X: samples stacked row-wise (as a 2D numpy array)
        :param y: labels of samples (as either list or numpy array)
        :param demo_mode: a boolean, if False uses input training samples, if True generates random 2D samples
        :param report_steps: a boolean, if true reports the steps done in PSA
        :return: ranks of the training samples within each class --> the first and last elements are respectively the indices of samples having the best and worst ranks. --> a list whose every element is for a class, and in every element, ranks of samples are stacked row-wise
        """
        # ------ Parameters:
        add_noisy_samples_if_not_necessary = self.add_noisy_samples_if_not_necessary
        number_of_added_noisy_samples = self.number_of_added_noisy_samples
        number_of_selected_samples_in_group = self.number_of_selected_samples_in_group
        number_of_iterations_of_RANSAC = self.number_of_iterations_of_RANSAC
        # ------ Settings:
        save_figures = demo_mode
        show_images = False
        do_plot_samples = demo_mode
        do_plot_qualified_samples = demo_mode
        do_plot_ranked_qualified_samples = demo_mode
        do_plot_ranked_samples = demo_mode
        do_plot_ranked_samples_without_noise = demo_mode
        # ------ PCA on data:
        if Apply_PCA:
            if report_steps: print('Applying PCA on data...')
            total_number_samples = X.shape[0]
            n_components = min(n_components, total_number_samples) # because after PCA, dimension <= number of samples ===> see: I googled: pca can dimensions be more than observations ==> https://stats.stackexchange.com/questions/28909/pca-when-the-dimensionality-is-greater-than-the-number-of-samples
            pca = decomposition.PCA(n_components=n_components)
            pca.fit(X=X)
            X = pca.transform(X=X)
        # ------ generate demo samples (if is demo_mode):
        if demo_mode:
            if HavingClasses:
                number_of_classes = self.number_of_classes
                dimension_of_data = self.dimension_of_data
                number_of_samples_of_each_class = self.number_of_samples_of_each_class
                class_samples_list = [None] * number_of_classes   # class_samples_list: a list whose every element is training samples of a class. In every element of the list, training samples are stacked row-wise.
                for class_index in range(number_of_classes):
                    mean = np.zeros(dimension_of_data)
                    cov = np.eye(dimension_of_data)
                    for d in range(dimension_of_data):
                        mean[d] = random.uniform(0+3*class_index, 1+3*class_index)
                        cov[d,d] = random.uniform(0+class_index, 5+class_index)
                    class_samples_list[class_index] = self.Create_Gaussian_samples(mean=mean, cov=cov, size=number_of_samples_of_each_class[class_index])
            else:
                number_of_classes = 1
                dimension_of_data = self.dimension_of_data
                number_of_samples_of_each_class = self.number_of_samples_of_each_class
                class_samples_list = [None] * number_of_classes   # class_samples_list: a list whose every element is training samples of a class. In every element of the list, training samples are stacked row-wise.
                for class_index in range(number_of_classes):
                    mean = np.zeros(dimension_of_data)
                    cov = np.eye(dimension_of_data)
                    for d in range(dimension_of_data):
                        mean[d] = random.uniform(0+3*class_index, 1+3*class_index)
                        cov[d,d] = random.uniform(0+class_index, 5+class_index)
                    class_samples_list[class_index] = self.Create_Gaussian_samples(mean=mean, cov=cov, size=number_of_samples_of_each_class[class_index])
        else:
            if HavingClasses:
                labels = list(set(y))  # set() removes redundant numbers in y and sorts them
                number_of_classes = len(labels)
                total_number_of_samples = len(y)
                dimension_of_data = X.shape[1]
                class_samples_list = [None] * number_of_classes   # class_samples_list: a list whose every element is training samples of a class. In every element of the list, training samples are stacked row-wise.
                for class_index in range(number_of_classes):
                    class_samples_list[class_index] = np.empty([0, dimension_of_data])
                for sample_index in range(total_number_of_samples):
                    for class_index in range(number_of_classes):
                        if y[sample_index] == labels[class_index]:
                            class_samples_list[class_index] = np.vstack([class_samples_list[class_index], X[sample_index, :]])
            else:
                number_of_classes = 1
                total_number_of_samples = X.shape[0]
                dimension_of_data = X.shape[1]
                class_samples_list = [None] * number_of_classes   # class_samples_list: a list whose every element is training samples of a class. In every element of the list, training samples are stacked row-wise.
                class_samples_list[0] = X
        # ------ characteristics of data:
        number_of_classes = len(class_samples_list)
        dimension_of_data = class_samples_list[0].shape[1]
        original_number_of_samples_of_each_class = [None] * number_of_classes
        for class_index in range(number_of_classes):
            original_number_of_samples_of_each_class[class_index] = class_samples_list[class_index].shape[0]
        # ------ check for being valid data:
        if (number_of_selected_samples_in_group < dimension_of_data) or (number_of_selected_samples_in_group > max(dimension_of_data, min(original_number_of_samples_of_each_class))):
            print('Technical Error: should be: dimension <= number_of_selected_samples_in_group <= max(dimension, min(number of samples of each class)). '
                  '\nIn other words: if all classes have population >= dimension, number_of_selected_samples_in_group should be <= min(number of samples of each class) '
                  '\nOtherwise: number_of_selected_samples_in_group should be = dimension'
                  '\nPlease select another value for variable number_of_selected_samples_in_group')
            return -1
        if report_steps: print('Input training data is read...')
        # ----- Creating random colors for plots (if is demo_mode):
        if demo_mode:
            color_of_plot = [None] * number_of_classes
            for class_index in range(number_of_classes):
                color_of_plot[class_index] = np.random.rand(1,3)
        # ----- add noisy samples if necessary:
        noisy_samples_are_added = [None] * number_of_classes
        number_of_samples_of_each_class = [None] * number_of_classes
        noise_labels = [None] * number_of_classes
        for class_index in range(number_of_classes):
            noisy_samples_are_added[class_index] = False
            if original_number_of_samples_of_each_class[class_index] < dimension_of_data:
                noisy_samples_are_added[class_index] = True
                number_of_added_noisy_samples_because_of_dimension = dimension_of_data - original_number_of_samples_of_each_class[class_index]
                class_samples_list[class_index], noise_labels[class_index] = PSA.Add_Gaussian_noise(self=self, X=class_samples_list[class_index], number_of_added_samples=number_of_added_noisy_samples_because_of_dimension)
            elif add_noisy_samples_if_not_necessary:
                noisy_samples_are_added[class_index] = True
                class_samples_list[class_index], noise_labels[class_index] = PSA.Add_Gaussian_noise(self=self, X=class_samples_list[class_index], number_of_added_samples=number_of_added_noisy_samples)
            number_of_samples_of_each_class[class_index] = class_samples_list[class_index].shape[0]
            if report_steps:
                if noisy_samples_are_added[class_index] is True:
                    print('Added ' + str(number_of_samples_of_each_class[class_index] - original_number_of_samples_of_each_class[class_index]) + ' noisy samples to class ' + str(class_index) + '...')
        # ------ Plot samples:
        if do_plot_samples and dimension_of_data==2:
            plot_samples(self, class_samples_list=class_samples_list, noise_labels=noise_labels, color_of_plot=color_of_plot, save_figures=save_figures, show_images=show_images)
        # ----- Group ranking:
        if report_steps: print('Start of group ranking...')
        group_sample_indices = [None] * number_of_classes
        for class_index in range(number_of_classes):
            groupRanking = GroupRanking(number_of_iterations_of_RANSAC=number_of_iterations_of_RANSAC)
            if not for_regression_task:
                group_sample_indices[class_index], best_group_score, beta_all, beta_of_best_group = groupRanking.find_best_group(number_of_selected_samples_in_group=number_of_selected_samples_in_group, class_index=class_index, class_samples_list=class_samples_list, simplified_version=True, HavingClasses=HavingClasses)
            else:
                group_sample_indices[class_index], best_group_score, beta_all, beta_of_best_group = groupRanking.find_best_group_for_regression(number_of_selected_samples_in_group=number_of_selected_samples_in_group, class_index=class_index, class_samples_list=class_samples_list, Y=y, simplified_version=True, HavingClasses=HavingClasses)
            if report_steps: print('group ranking in class ' + str(class_index) + ' is done.')
        if report_steps: print('End of group ranking...')
        # ------ Plot qualified (grouped) samples:
        if do_plot_qualified_samples and dimension_of_data==2:
            plot_group_ranking(self, class_samples_list, group_sample_indices, color_of_plot, save_figures, show_images)
        # ----- Individual ranking in selected groups:
        if report_steps: print('Start of individual ranking of qualified samples...')
        Individual_ranks = [None] * number_of_classes
        for class_index in range(number_of_classes):
            Individual_ranks[class_index] = np.zeros((number_of_samples_of_each_class[class_index], 1))
        individualRankingInGroups = IndividualRankingInGroups()
        for class_index in range(number_of_classes):
            samples_of_class = class_samples_list[class_index]
            number_of_samples_of_this_class = samples_of_class.shape[0]
            mask = np.in1d(range(number_of_samples_of_this_class), group_sample_indices[class_index])  # https://stackoverflow.com/questions/27303225/numpy-vstack-empty-initialization
            grouped_samples = samples_of_class[mask,:]
            ranks_among_group_samples, individual_scores = individualRankingInGroups.individual_rank_of_qualified_samples(class_index=class_index, grouped_samples=grouped_samples, class_samples_list=class_samples_list, simplified_version=True, HavingClasses=HavingClasses)
            for index in range(len(ranks_among_group_samples)):
                rank = int(ranks_among_group_samples[index])
                sample_index_among_all_samples_of_class = group_sample_indices[class_index][rank]
                Individual_ranks[class_index][index] = sample_index_among_all_samples_of_class
        if report_steps: print('End of individual ranking of qualified samples...')
        # ------ Plot ranked qualified (grouped) samples:
        if do_plot_ranked_qualified_samples and dimension_of_data==2:
            plot_individual_ranking_in_groups(self, class_samples_list, group_sample_indices, Individual_ranks, color_of_plot, save_figures, show_images)
        # ------ Individual ranking of unqualified samples:
        if report_steps: print('Start of individual ranking of unqualified samples...')
        individualRankingInUnqualifieds = IndividualRankingInUnqualifieds()
        for class_index in range(number_of_classes):
            ranks_among_unqualified_samples, individual_scores, Individual_ranks = individualRankingInUnqualifieds.individual_rank_of_unqualified_samples(class_index=class_index, class_samples_list=class_samples_list, Individual_ranks=Individual_ranks, group_sample_indices=group_sample_indices, HavingClasses=HavingClasses)
        if report_steps: print('End of individual ranking of unqualified samples...')
        # ------ Plot ranked samples:
        if do_plot_ranked_samples and dimension_of_data==2:
            plot_individual_ranking_in_all(self, class_samples_list, group_sample_indices, Individual_ranks, color_of_plot, save_figures, show_images)
        # ------ remove the noisy samples:
        pure_Individual_ranks = [None] * number_of_classes
        class_samples_list_excludingNoisySamples = [None] * number_of_classes
        for class_index in range(number_of_classes):
            if noisy_samples_are_added[class_index] is True:
                # ------ remove the ranks of noise sample from individual ranks:
                number_of_pure_samples = original_number_of_samples_of_each_class[class_index]
                pure_Individual_ranks[class_index] = np.zeros((number_of_pure_samples, 1))
                counter = 0
                for i in range(Individual_ranks[class_index].shape[0]):
                    if noise_labels[class_index][i] == False:
                        pure_Individual_ranks[class_index][counter] = Individual_ranks[class_index][i]
                        counter += 1
                # ------ remove the bias of ranks because of ranks of noisy samples which are now removed:
                pure_Individual_ranks_BiasesRemoved = pure_Individual_ranks[class_index].copy()
                for i in range(len(pure_Individual_ranks[class_index])): # iteration on elements of pure_Individual_ranks of a class
                    counter = 0
                    if int(pure_Individual_ranks[class_index][i]) != 0:
                        for j in range(int(pure_Individual_ranks[class_index][i])):  # iteration on ranks before it
                            if j not in pure_Individual_ranks[class_index][:]:  # means: if rank j was assigned to a noisy sample which is now removed
                                counter += 1
                    pure_Individual_ranks_BiasesRemoved[i] -= counter
                pure_Individual_ranks[class_index] = pure_Individual_ranks_BiasesRemoved
                # ------ remove the noisy samples from class_samples_list:
                pure_samples = np.zeros((number_of_pure_samples, dimension_of_data))
                pure_sample_index = 0
                for sample_index in range(class_samples_list[class_index].shape[0]):
                    if noise_labels[class_index][sample_index] == False:
                        sample = class_samples_list[class_index][sample_index,:]
                        pure_samples[pure_sample_index, :] = sample
                        pure_sample_index += 1
                class_samples_list_excludingNoisySamples[class_index] = pure_samples
                if report_steps: print('Excluding noisy samples is done in class ' + str(class_index) + '...')
            else:
                class_samples_list_excludingNoisySamples[class_index] = class_samples_list[class_index]
                pure_Individual_ranks[class_index] = Individual_ranks[class_index]
        # ------ Plot ranked samples:
        if do_plot_ranked_samples_without_noise and dimension_of_data==2:
            plot_final_ranked_samples(self, class_samples_list_excludingNoisySamples, pure_Individual_ranks, color_of_plot, save_figures, show_images)
        # ------ save the individual ranks:
        if self.save_ranks is True:
            PSA.save_variable(self, pure_Individual_ranks, 'ranks', path_to_save='./PSA_outputs/')
        # ------ save the individual ranks in text files:
        if self.save_ranks is True:
            path_to_save='./PSA_outputs/'
            for class_index in range(number_of_classes):
                f = open(path_to_save + 'ranks_in_class' + str(class_index) + '.txt', 'w')
                for line in pure_Individual_ranks[class_index]:
                    f.write(str(int(line)))
                    f.write('\n')
                f.close()
            if report_steps: print('Results are saved...')
        # ------ return:
        ranks = pure_Individual_ranks
        return ranks

    def sort_samples_classWise(self, ranks, X, Y, for_classification):
        # X --> rows are samples and columns are features
        X = X.T
        n_dimensions = X.shape[0]
        if for_classification is True:
            X_separated_classes = self.separate_samples_of_classes(X=X.T, y=Y.ravel())
            n_classes = len(X_separated_classes)
            X_final_sorted = np.empty((n_dimensions, 0))
            Y_final_sorted = np.empty(((Y).shape[0], 0))
            scores_sorted = np.empty((1, 0))
            for class_index in range(n_classes):
                samples_of_class = X_separated_classes[class_index].T
                n_samples_of_class = samples_of_class.shape[1]
                labels_of_samples_of_class = np.ones((1, n_samples_of_class)) * class_index
                X_final_sorted_ofClass, Y_final_sorted_ofClass, scores_sorted_ofClass = self.sort_samples(ranks=ranks[class_index], X=samples_of_class.T, Y=labels_of_samples_of_class)
                X_final_sorted = np.column_stack((X_final_sorted, X_final_sorted_ofClass))
                Y_final_sorted = np.column_stack((Y_final_sorted, Y_final_sorted_ofClass))
                scores_sorted = np.column_stack((scores_sorted, scores_sorted_ofClass.reshape(1, -1)))
        else:
            X_final_sorted, Y_final_sorted, scores_sorted = self.sort_samples(ranks=ranks, X=X.T, Y=Y)
        return X_final_sorted, Y_final_sorted, scores_sorted

    def sort_samples(self, ranks, X, Y):
        # X --> rows are samples and columns are features
        # output: X_sorted, Y_sorted, scores_sorted --> (rows are features, columns are samples)
        X = X.T
        n_dimensions = X.shape[0]
        # convert ranks to scores:
        ranks = np.asarray(ranks)
        ranks = ranks.reshape((1, -1))
        scores = -1 * ranks
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

    def separate_samples_of_classes(self, X, y):  # it does not change the order of the samples within every class
        # X --> rows: samples, columns: features
        # return X_separated_classes --> each element of list --> rows: samples, columns: features
        y = np.asarray(y)
        y = y.reshape((-1, 1)).ravel()
        labels_of_classes = sorted(set(y.ravel().tolist()))
        n_samples = X.shape[0]
        n_dimensions = X.shape[1]
        n_classes = len(labels_of_classes)
        X_separated_classes = [np.empty((0, n_dimensions))] * n_classes
        for class_index in range(n_classes):
            X_separated_classes[class_index] = X[y == labels_of_classes[class_index], :]
        return X_separated_classes

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

    def sort_samples_according_to_ranks(self, X, y, ranks):
        labels = list(set(y))  # set() removes redundant numbers in y and sorts them
        number_of_classes = len(labels)
        total_number_of_samples = len(y)
        dimension_of_data = X.shape[1]
        # creating class_samples_list out of X and y:
        class_samples_list = [None] * number_of_classes   # class_samples_list: a list whose every element is training samples of a class. In every element of the list, training samples are stacked row-wise.
        for class_index in range(number_of_classes):
            class_samples_list[class_index] = np.empty([0, dimension_of_data])
        for sample_index in range(total_number_of_samples):
            for class_index in range(number_of_classes):
                if y[sample_index] == labels[class_index]:
                    class_samples_list[class_index] = np.vstack([class_samples_list[class_index], X[sample_index, :]])
        # sorting according to ranks:
        sorted_samples = [None] * number_of_classes   # sorted_samples: a list whose every element contains sorted training samples of a class. In every element of the list, sorted training samples are stacked row-wise.
        for class_index in range(number_of_classes):
            sorted_samples[class_index] = np.zeros(class_samples_list[class_index].shape)
        for class_index in range(number_of_classes):
            samples_of_class = class_samples_list[class_index]
            number_of_samples_of_class = samples_of_class.shape[0]
            for sample_index in range(number_of_samples_of_class):
                rank_among_all_samples_of_class = int(np.where(ranks[class_index][:] == sample_index)[0])  # find place of sample_index in ranks[class_index][:]
                sorted_samples[class_index][rank_among_all_samples_of_class, :] = class_samples_list[class_index][sample_index, :]
        return sorted_samples

    def reduce_data(self, sorted_samples, n_samples):
        number_of_classes = len(sorted_samples)
        dimension_of_data = sorted_samples[0].shape[1]
        X_reducedData = np.empty([0, dimension_of_data])
        y_reducedData = []
        for class_index in range(number_of_classes):
            sorted_samples_of_class = sorted_samples[class_index]
            number_of_sampled_data = n_samples[class_index]
            X_reducedData = np.vstack([X_reducedData, sorted_samples_of_class[:number_of_sampled_data, :]])
            y_reducedData.extend([class_index] * number_of_sampled_data)
        y_reducedData = np.asarray(y_reducedData)
        X = X_reducedData
        y = y_reducedData
        return X, y


    # ------ functions:
    def save_variable(self, variable, name_of_variable, path_to_save='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.pckl'
        f = open(file_address, 'wb')
        pickle.dump(variable, f)
        f.close()

    def load_variable(self, name_of_variable, path_to_save='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        file_address = path_to_save + name_of_variable + '.pckl'
        f = open(file_address, 'rb')
        variable = pickle.load(f)
        f.close()
        return variable

    def Add_Gaussian_noise(self, X, number_of_added_samples):
        covariance_of_data_distribution = np.cov(X.T)
        mean_of_Gaussian = X.mean(axis=0)  # mean of data samples
        noisy_samples = PSA.Create_Gaussian_samples(self, mean=mean_of_Gaussian, cov=covariance_of_data_distribution, size=number_of_added_samples)
        number_of_samples = X.shape[0]
        X_expanded = np.vstack([X, noisy_samples])
        new_number_of_samples = number_of_samples + number_of_added_samples
        noise_labels = np.zeros(new_number_of_samples)
        for sample_index in range(new_number_of_samples):
            if sample_index < number_of_samples:
                noise_labels[sample_index] = False
            else:
                noise_labels[sample_index] = True
        return X_expanded, noise_labels

    def Create_Gaussian_samples(self, mean=np.zeros(2), cov=np.eye(2), size=5):
        # https://stackoverflow.com/questions/34932499/draw-multiple-samples-using-numpy-random-multivariate-normalmean-cov-size
        return np.random.multivariate_normal(mean=mean, cov=cov, size=size)

