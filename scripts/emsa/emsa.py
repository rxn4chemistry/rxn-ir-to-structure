import numpy as np


class EMSA:
    """
    Extended Multiplicative Signal Augmentation
    Generates balanced batches of augmentated spectra
    """

    def __init__(self, std_of_params, wavenumbers, reference, order=2):
        """
        :param std_of_params: array of length (order+2), which
        :param reference: reference spectrum that was used in EMSC model
        :param order: order of emsc
        contains the std for each coefficient
        """
        self.order = order
        self.std_of_params = std_of_params
        self.ref = reference
        self.X = None
        self.A = None
        self.__create_x_and_a(wavenumbers)

    def generator(self, spectra, labels,
                  equalize_subsampling=False, shuffle=True,
                  batch_size=32):
        """ generates batches of transformed spectra"""
        spectra = np.asarray(spectra)
        labels = np.asarray(labels)

        if self.std_of_params is None:
            coefs = np.dot(self.A, spectra.T)
            self.std_of_params = coefs.std(axis=1)

        if equalize_subsampling:
            indexes = self.__rearrange_spectra(labels)
        else:
            indexes = np.arange(len(spectra))

        cur = 0
        while True:
            if shuffle:
                si = indexes[np.random.randint(len(indexes),
                                               size=batch_size)]
            else:
                si = indexes.take(range(cur, cur + batch_size),
                                  mode='wrap')
                cur += batch_size

            yield self.batch_transform(spectra[si]), labels[si]

    def __rearrange_spectra(self, labels):
        """ returns indexes of data rearranged in the way of 'balance'"""
        classes = np.unique(labels, axis=0)

        if len(labels.shape) == 2:
            grouped = [np.where(np.all(labels == l, axis=1))[0]
                       for l in classes]
        else:
            grouped = [np.where(labels == l)[0] for l in classes]
        iters_cnt = max([len(g) for g in grouped])

        indexes = []
        for i in range(iters_cnt):
            for g in grouped:
                # take cyclic sample from group
                indexes.append(np.take(g, i, mode='wrap'))

        return np.array(indexes)

    def __create_x_and_a(self, wavenumbers):
        """
        Builds X matrix from spectra in such way that columns go as
        reference w^0 w^1 w^2 ... w^n, what corresponds to coefficients
        b, a, d, e, ...
        and caches the solution self.A = (X^T*X)^(-1)*X^T
        :param spectra:
        :param wavenumbers:
        :return: nothing, but creates two self.X and self.A
        """
        # squeeze wavenumbers to approx. range [-1; 1]
        # use if else to support uint types
        if wavenumbers[0] > wavenumbers[-1]:
            rng = wavenumbers[0] - wavenumbers[-1]
        else:
            rng = wavenumbers[-1] - wavenumbers[0]
        half_rng = rng / 2
        normalized_wns = (wavenumbers - np.mean(wavenumbers)) / half_rng

        self.polynomial_columns = [np.ones_like(wavenumbers)]
        for j in range(1, self.order + 1):
            self.polynomial_columns.append(normalized_wns ** j)

        self.X = np.stack((self.ref, *self.polynomial_columns), axis=1)
        self.A = np.dot(np.linalg.pinv(np.dot(self.X.T, self.X)), self.X.T)

    def batch_transform(self, spectra):
        spectra_columns = spectra.T

        # b, a, d, e, ...
        coefs = np.dot(self.A, spectra_columns)
        residues = spectra_columns - np.dot(self.X, coefs)

        new_coefs = coefs.copy()

        # wiggle coefficients
        for i in range(len(coefs)):
            new_coefs[i] += np.random.normal(0,
                                             self.std_of_params[i],
                                             len(spectra))

        # Fix if multiplication parameter sampled negative
        mask = new_coefs[0] <= 0
        if np.any(mask):
            # resample multiplication parameter to be positive
            n_resamples = mask.sum()
            new_coefs[0][mask] = np.random.uniform(0, coefs[0][mask],
                                                   n_resamples)

        return (np.dot(self.X, new_coefs) + residues * new_coefs[0] / coefs[0]).T
