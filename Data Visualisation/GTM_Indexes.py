from GTM import GTM
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


class GTMIndexes(GTM):
    def __init__(self, input_data=sp.rand(100, 3), rbf_number=25, rbf_width=1, regularization=1,
                 latent_space_size=3600, iterations=100):

        # Initialization of the procedure to evaluate GTM's performance
        GTM.__init__(self, input_data=input_data, rbf_number=rbf_number, rbf_width=rbf_width,
                     regularization=regularization, latent_space_size=latent_space_size, iterations=iterations)
        self.input_data_remapped = np.zeros((self.input_data.shape[0], self.input_data.shape[1]))
        self.gtm_new_distance = np.zeros((self.latent_space_size, self.input_data.shape[0]))
        self.gtm_new_responsibility = np.zeros((self.latent_space_size, self.input_data.shape[0]))

    def gtm_scaling(self, data=None):
        """ Scaling and centering data using training data's mean and standard deviation

        :param data: new data set
        :return: data: scaled data set
        """
        if data is None:
            data = self.input_data
        mean = np.mean(self.input_data, axis=0)
        std = np.std(self.input_data, axis=0)
        data = (data - mean)/std
        data = np.nan_to_num(data)

        return data

    def gtm_new_responsibilities(self, w, beta, data=None):
        """ Calculating responsibilities for a new data set

        :param w: optimal weight matrix
        :param beta: optimal scalar value of the inverse variance common to all components of the mixture
        :param data: new data set
        """
        if data is None:
            data = self.centered_input_data
        self.gtm_new_distance = cdist(np.dot(self.fi, w), data, 'sqeuclidean')
        dist_corr = np.minimum((self.gtm_new_distance.max(axis=0) + self.gtm_new_distance.min(axis=0)) / 2,
                               self.gtm_new_distance.min(axis=0) + (700 * 2 / beta))
        for i in range(0, self.gtm_new_distance.shape[1]):
            self.gtm_new_distance[:, i] = self.gtm_new_distance[:, i] - dist_corr[i]
        self.gtm_new_responsibility = np.exp((-beta / 2) * self.gtm_new_distance)
        responsibility_sum = np.sum(self.gtm_new_responsibility, 0)
        self.gtm_new_responsibility = self.gtm_new_responsibility / np.transpose(responsibility_sum[:, None])

    def gtm_remap(self, w, beta, data=None):
        """ Calculate remapped input data to original hyperspace

        :param w: optimal weight matrix
        :param beta: optimal scalar value of the inverse variance common to all components of the mixture
        :param data: new data set
        """
        if data is None:
            data = self.centered_input_data
        self.gtm_new_responsibilities(w, beta, data)
        self.input_data_remapped = np.dot(np.transpose(self.gtm_new_responsibility), np.dot(self.fi, w))
        self.input_data_remapped = np.nan_to_num(self.input_data_remapped)

    def gtm_r2(self, w, beta, data=None):
        """ r2 score based on remapped data

        :param w: optimal weight matrix
        :param beta: optimal scalar value of the inverse variance common to all components of the mixture
        :param data: new data set
        :return: r2: coefficient of determination regression score
        """
        if data is None:
            data = self.centered_input_data
        data = self.gtm_scaling(data)
        self.gtm_remap(w, beta, data)
        r2 = r2_score(data, self.input_data_remapped)
        return r2

    def gtm_rmse(self, w, beta, data=None):
        """ r2 score based on remapped data

        :param w: optimal weight matrix
        :param beta: optimal scalar value of the inverse variance common to all components of the mixture
        :param data: new data set
        :return: r2: coefficient of determination regression score
        """
        if data is None:
            data = self.centered_input_data
        data = self.gtm_scaling(data)
        self.gtm_remap(w, beta, data)
        rmse = mean_squared_error(data, self.input_data_remapped)
        return rmse

    def gtm_distance_index(self, w, beta, data=None):
        """ r2 score based on distance rankings for each sample

        :param w: optimal weight matrix
        :param beta: optimal scalar value of the inverse variance common to all components of the mixture
        :param data: new data set
        :return: r2: Spearman's rank correlation index
        """

        if data is None:
            data = self.centered_input_data
        data = self.gtm_scaling(data)
        mean_matrix = self.gtm_new_mean(w, beta, data)
        distance_original = squareform(pdist(data))
        distance_remapped = squareform(pdist(mean_matrix))
        r2_vector = []
        for i in range(0, distance_original.shape[0]):
            idx_original = np.argsort(distance_original[:, i])
            idx_remapped = np.argsort(distance_remapped[:, i])
            # Spearman's rank correlation coefficient
            r2_vector.append(1.0 - 6.0/(float(self.input_data.shape[0]**3-self.input_data.shape[0]))*np.sum(
                (idx_original-idx_remapped)**2))
        r2 = np.mean(r2_vector)
        return r2

    def gtm_r2_neighbors(self, w, beta, data=None):
        """

        :param w: optimal weight matrix
        :param beta: optimal scalar value of the inverse variance common to all components of the mixture
        :param data: new data set
        :return: r2: coefficient of determination regression score
        """
        if data is None:
            data = self.centered_input_data
        data = self.gtm_scaling(data)
        distance_original = squareform(pdist(data, 'euclidean'))
        idx_matrix = np.argsort(distance_original)
        neighbor_input_data = np.zeros((self.input_data.shape[0], self.input_data.shape[1]))
        for i in range(0, distance_original.shape[0]):
            neighbor_input_data[i, :] = np.mean(data[idx_matrix[i, 1:4], :], axis=0)
        self.gtm_remap(w, beta, neighbor_input_data)
        r2 = r2_score(neighbor_input_data, self.input_data_remapped)
        return r2

    def gtm_midpoint(self, data=None):
        """ Generation of midpoint data for GTM optimization

        :param data: new data set
        :return: xm: midpoint values
        """
        if data is None:
            data = self.input_data
        idx1 = np.random.randint(data.shape[0], size=(data.shape[0]*5)).tolist()
        idx2 = np.random.randint(data.shape[0], size=(data.shape[0]*5)).tolist()
        xm1 = data[idx1, :]
        xm2 = data[idx2, :]
        xm = (xm1+xm2)/2.0
        b = np.ascontiguousarray(xm).view(np.dtype((np.void, xm.dtype.itemsize * xm.shape[1])))
        _, idx = np.unique(b, return_index=True)
        xm = xm[idx]
        return xm

    def gtm_midpoint_neighbors(self, data=None, neighbors=None):
        """ Generation of midpoint data for GTM optimization considering sample neighborhood

        :param data: new data set
        :param neighbors: how many neighbors are being considered for generation
        :return: xm: midpoint values
        """
        if data is None:
            data = self.input_data
        if neighbors is None:
            neighbors = data.shape[0]
        distance_original = squareform(pdist(data, 'euclidean'))
        idx_matrix = np.argsort(distance_original)
        for i in range(0, data.shape[0]):
            xm1 = np.matlib.repmat(data[i, :], neighbors, 1)
            xm2 = data[idx_matrix[i, 1:11], :]
            xmiter = (xm1+xm2)/2.0
            if i == 0:
                xm = xmiter
            else:
                xm = np.row_stack((xm, xmiter))
        b = np.ascontiguousarray(xm).view(np.dtype((np.void, xm.dtype.itemsize * xm.shape[1])))
        _, idx = np.unique(b, return_index=True)
        xm = xm[idx]
        return xm

    def gtm_r2_initialization(self):
        """ Generation of GTM components used with a 2D latent space """
        # Create GTM latent space grid vectors
        latent_space_dimension = np.sqrt(self.latent_space_size)
        self.z = self.gtm_rectangular(latent_space_dimension)
        self.z = self.z[::-1, ::-1]
        # Create GTM latent rbf grid vectors
        rbf_dimension = np.sqrt(self.rbf_number)
        mu = self.gtm_rectangular(rbf_dimension)
        mu = mu * rbf_dimension / (rbf_dimension - 1)
        mu = mu[::-1, ::-1]
        # Calculate the spread of the basis functions
        sigma = self.rbf_width * np.abs(mu[1, 0] - mu[1, 1])
        # Calculate the activations of the hidden unit when fed the latent variable samples
        self.fi = self.gtm_gaussian_basis_functions(mu, sigma)

    def gtm_new_mean(self, w, beta, data=None):
        """ Find mean probability density values for each sample in the latent space
        :param w: optimal weight matrix
        :param beta: optimal scalar value of the inverse variance common to all components of the mixture
        :param data: new data set
        """
        if data is None:
            data = self.centered_input_data
        data = self.gtm_scaling(data)
        self.gtm_distance = cdist(np.dot(self.fi, w), data, 'sqeuclidean')
        self.gtm_responsibilities(beta)
        means = np.dot(np.transpose(self.gtm_responsibility), np.transpose(self.z))
        return means
