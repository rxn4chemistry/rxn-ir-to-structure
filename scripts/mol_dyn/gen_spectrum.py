# Adapted from Efrem Braun https://github.com/EfremBraun/calc-ir-spectra-from-lammps, doi.org/10.5281/zenodo.154672

import os

import numpy as np
from scipy import fftpack, signal


def gen_spectrum(save_dir: str) -> None:
    # Inputs
    autocorrelation_option = 1  # 1 to calculate it, 2 to load a pre-calculated one
    T = 300.0  # K
    # and at 1% of the trajectory all datapoints have at least 100 non-ish-correlated counts
    input_dipole_file = os.path.join(save_dir, "dipole.txt")
    output_autocorrelation_file = os.path.join(save_dir, "autocorr.txt")

    # Constants
    boltz = 1.38064852e-23  # m^2 kg s^-2 K^-1
    lightspeed = 299792458.0  # m s^-1
    reduced_planck = 1.05457180013e-34  # kg m^2 s^-1

    nq = 16678
    data_points = nq / 2

    ###############################################################################
    # Get autocorrelation function
    ###############################################################################
    # Calculate autocorrelation function
    if autocorrelation_option == 1:
        # Load data
        time, dipole_x, dipole_y, dipole_z = np.loadtxt(
            input_dipole_file, skiprows=2, usecols=(1, 2, 3, 4), unpack=True
        )

        # Do calculation
        # Note that this method of calculating an autocorrelation function is very fast, but it can be difficult to follow.
        # For readability, I've presented a more straightforward (but much, much slower) method in the commented block below.
        # Shift the array
        if len(time) % 2 == 0:
            dipole_x_shifted = np.zeros(len(time) * 2)
            dipole_y_shifted = np.zeros(len(time) * 2)
            dipole_z_shifted = np.zeros(len(time) * 2)
        else:
            dipole_x_shifted = np.zeros(len(time) * 2 - 1)
            dipole_y_shifted = np.zeros(len(time) * 2 - 1)
            dipole_z_shifted = np.zeros(len(time) * 2 - 1)
        dipole_x_shifted[len(time) // 2 : len(time) // 2 + len(time)] = dipole_x
        dipole_y_shifted[len(time) // 2 : len(time) // 2 + len(time)] = dipole_y
        dipole_z_shifted[len(time) // 2 : len(time) // 2 + len(time)] = dipole_z
        # Convolute the shifted array with the flipped array, which is equivalent to performing a correlation
        autocorr_x_full = signal.fftconvolve(
            dipole_x_shifted, dipole_x[::-1], mode="same"
        )[(-len(time)) :] / np.arange(len(time), 0, -1)
        autocorr_y_full = signal.fftconvolve(
            dipole_y_shifted, dipole_y[::-1], mode="same"
        )[(-len(time)) :] / np.arange(len(time), 0, -1)
        autocorr_z_full = signal.fftconvolve(
            dipole_z_shifted, dipole_z[::-1], mode="same"
        )[(-len(time)) :] / np.arange(len(time), 0, -1)
        autocorr_full = autocorr_x_full + autocorr_y_full + autocorr_z_full
        # Truncate the autocorrelation array
        autocorr = autocorr_full[: int(data_points)]

        # Save data
        np.savetxt(
            output_autocorrelation_file,
            np.column_stack((time[: len(autocorr)], autocorr)),
            header="Time(fs) Autocorrelation(e*Ang)",
        )

    # Load pre-calculated autocorrelation function
    elif autocorrelation_option == 2:
        time, autocorr = np.loadtxt(
            output_autocorrelation_file, skiprows=1, unpack=True
        )

    else:
        print("Not a valid option for 'autocorrelation_option'.")

    timestep = (1) * 1.0e-15  # converts time from femtoseconds to seconds

    ###############################################################################
    # Calculate spectra
    # Note that intensities are relative, and so can be multiplied by a constant to compare to experiment.
    ###############################################################################
    # Calculate the FFTs of autocorrelation functions
    lineshape = fftpack.dct(autocorr, type=1)[1:]
    lineshape_frequencies = np.linspace(0, 0.5 / timestep, len(autocorr))[1:]
    lineshape_frequencies_wn = lineshape_frequencies / (
        100.0 * lightspeed
    )  # converts to wavenumbers (cm^-1)

    # Calculate spectra
    field_description = lineshape_frequencies * (
        1.0 - np.exp(-reduced_planck * lineshape_frequencies / (boltz * T))
    )
    quantum_correction = lineshape_frequencies / (
        1.0 - np.exp(-reduced_planck * lineshape_frequencies / (boltz * T))
    )
    # quantum correction per doi.org/10.1021/jp034788u. Other options are possible, see doi.org/10.1063/1.441739 and doi.org/10.1080/00268978500102801.
    spectra = lineshape * field_description
    spectra_qm = spectra * quantum_correction

    # Save data
    save_path = os.path.join(save_dir, "IR-data.csv")
    np.savetxt(
        save_path,
        np.column_stack(
            (
                lineshape_frequencies_wn,
                lineshape,
                field_description,
                quantum_correction,
                spectra,
                spectra_qm,
            )
        ),
        header="Frequency(cm^-1), Lineshape, Field_description, Quantum_correction, Spectra, Spectra_qm",
        delimiter=",",
    )
