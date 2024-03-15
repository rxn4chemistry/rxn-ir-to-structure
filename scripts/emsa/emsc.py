from typing import Union as U, Tuple as T

import numpy as np


def emsc(spectra: np.ndarray, wavenumbers: np.ndarray, order: int = 2,
         reference: np.ndarray = None,
         constituents: np.ndarray = None,
         return_coefs: bool = False) -> U[np.ndarray, T[np.ndarray, np.ndarray]]:
    """
    Preprocess all spectra with EMSC
    :param spectra: ndarray of shape [n_samples, n_channels]
    :param wavenumbers: ndarray of shape [n_channels]
    :param order: order of polynomial
    :param reference: reference spectrum
    :param constituents: ndarray of shape [n_consituents, n_channels]
    Except constituents it can also take orthogonal vectors,
    for example from PCA.
    :param return_coefs: if True returns coefficients
    [n_samples, n_coeffs], where n_coeffs = 1 + len(costituents) + (order + 1).
    Order of returned coefficients:
    1) b*reference +                                    # reference coeff
    k) c_0*constituent[0] + ... + c_k*constituent[k] +  # constituents coeffs
    a_0 + a_1*w + a_2*w^2 + ...                         # polynomial coeffs
    :return: preprocessed spectra
    """
    if reference is None:
        reference = np.mean(spectra, axis=0)
    reference = reference[:, np.newaxis]

    # squeeze wavenumbers to approx. range [-1; 1]
    # use if else to support uint types
    if wavenumbers[0] > wavenumbers[-1]:
        rng = wavenumbers[0] - wavenumbers[-1]
    else:
        rng = wavenumbers[-1] - wavenumbers[0]
    half_rng = rng / 2
    normalized_wns = (wavenumbers - np.mean(wavenumbers)) / half_rng

    polynomial_columns = [np.ones(len(wavenumbers))]
    for j in range(1, order + 1):
        polynomial_columns.append(normalized_wns ** j)
    polynomial_columns = np.stack(polynomial_columns).T

    # spectrum = X*coefs + residues
    # least squares -> A = (X.T*X)^-1 * X.T; coefs = A * spectrum
    if constituents is None:
        columns = (reference, polynomial_columns)
    else:
        columns = (reference, constituents.T, polynomial_columns)

    X = np.concatenate(columns, axis=1)
    A = np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T)

    spectra_columns = spectra.T
    coefs = np.dot(A, spectra_columns)
    residues = spectra_columns - np.dot(X, coefs)

    preprocessed_spectra = (reference + residues/coefs[0]).T

    if return_coefs:
        return preprocessed_spectra, coefs.T

    return preprocessed_spectra
