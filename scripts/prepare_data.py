import os
from typing import Dict, List, Optional, Protocol, Tuple

import click
import numpy as np
import pandas as pd
import regex as re
import tqdm
import math
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import KFold, train_test_split
from functools import partial
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem

from emsa.emsa import EMSA
from emsa.emsc import emsc

class AugmentationCallable(Protocol):
    def __call__(self, x: np.ndarray, y: np.ndarray, /) -> List[np.ndarray]:
        ...


def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_pickle(data_path)


def split_smiles(smile: str) -> str:
    pattern_full = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

    regex = re.compile(pattern_full)
    tokens = [token for token in regex.findall(smile)]

    if smile != "".join(tokens):
        raise ValueError(
            "Tokenised smiles does not match original: {} {}".format(tokens, smile)
        )

    return " ".join(tokens)


def split_formula(formula: str) -> str:
    segments = re.findall(r"[A-Z][a-z]*\d*", formula)
    formula_split = [a for segment in segments for a in re.split(r"(\d+)", segment)]
    formula_split = list(filter(None, formula_split))

    if "".join(formula_split) != formula:
        raise ValueError(
            "Tokenised smiles does not match original: {} {}".format(
                formula_split, formula
            )
        )
    return " ".join(formula_split) + " | "


def norm_spectrum(
    spectrum: np.ndarray, bounds: Tuple[int, int] = (0, 99)
) -> np.ndarray:
    spectrum_norm = spectrum / max(spectrum) * bounds[1]
    spectrum_norm_int = spectrum_norm.astype(int)
    spectrum_norm_int = np.clip(spectrum_norm_int, *bounds)

    return spectrum_norm_int


def get_window(option: str, n_tokens: int, start: str) -> np.ndarray:
    if start == "450":
        start_val = 450
        end_val = 3850
    elif start == "550":
        start_val = 550
        end_val = 3850
    elif start == "800":
        start_val = 800
        end_val = 3980
    elif start == "N/A":
        start_val = 400
        end_val = 3980
    else:
        raise ValueError(f"Unknown option for start: {start}")
    
    if option == "full":
        return np.linspace(start_val, end_val, n_tokens)
    elif option == "fingerprint":
        return np.linspace(start_val, 2000, n_tokens)
    elif option == "umir":
        return np.linspace(2000, end_val, n_tokens)
    elif option == "merged":
        resolution = (2000 - start_val + 500) / n_tokens
        return np.concatenate(
            [
                np.arange(start_val, 2000, resolution),
                np.arange(2800, 3300 - resolution, resolution),
            ]
        )
    elif option == "remove_1800":
        resolution = (1800 - start_val + 500) / n_tokens
        return np.concatenate(
            [
                np.arange(start_val, 1800, resolution),
                np.arange(2800, 3300 - resolution, resolution),
            ]
        )
    else:
        raise KeyError(
            f"{option} not valid. Choose from [full, fingerprint, umir, merged]"
        )


def interpolate_spectrum(
    spectrum: np.ndarray,
    new_x: np.ndarray,
    orig_x: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    if orig_x is None:
        orig_x = np.arange(400, 3982, 2)

    intp = interpolate.interp1d(orig_x, spectrum)

    intp_spectrum = intp(new_x)
    intp_spectrum_norm = norm_spectrum(intp_spectrum)

    return [intp_spectrum_norm]


def augment_smooth(
    spectrum: np.ndarray, new_x: np.ndarray, sigmas: Optional[List[float]] = None
) -> List[np.ndarray]:
    if sigmas is None:
        sigmas = [0.75, 1.25]
    smoothed_spectra = list()
    for sigma in sigmas:
        smooth_spectrum = gaussian_filter1d(spectrum, sigma)
        smoothed_spectra.extend(interpolate_spectrum(smooth_spectrum, new_x))

    return smoothed_spectra


def augment_shift_horizontal(
    spectrum: np.ndarray, new_x: np.ndarray
) -> List[np.ndarray]:
    orig_x = np.arange(400, 3982, 4)
    set1 = spectrum[::2]
    set2 = np.concatenate([spectrum[1::2], np.reshape(set1[-1], 1)])

    aug_spec1 = interpolate_spectrum(set1, new_x, orig_x)[0]
    aug_spec2 = interpolate_spectrum(set2, new_x, orig_x)[0]
    return [aug_spec1, aug_spec2]


def augment_noise(
    spectrum: np.ndarray,
    new_x: np.ndarray,
    n_shifts: int = 2,
    shift_strength: float = 0.05,
) -> List[np.ndarray]:
    shift = max(spectrum) * shift_strength
    noised_spectra_vertical = list()

    for _ in range(n_shifts):
        noise = np.random.normal(0, 0.25, len(spectrum)) * shift
        noised_spectrum = spectrum + noise

        noised_spectrum = interpolate_spectrum(noised_spectrum, new_x)[0]

        noised_spectra_vertical.append(noised_spectrum)

    return noised_spectra_vertical

def augment_emsa(spectrum: np.ndarray, new_x: np.ndarray, emsa: EMSA) -> List[np.ndarray]:

    augmented_spectrum = emsa.batch_transform(np.array([spectrum]))[0]
    intp_spectrum = interpolate_spectrum(augmented_spectrum, new_x)[0]
    return [intp_spectrum]

def augment__offset(spectrum: np.ndarray, new_x: np.ndarray) -> List[np.ndarray]:
    std = np.std(spectrum)
    offset = std * np.random.normal(loc=0.2, scale=0.05)
    offset_spectrum = spectrum + offset
    offset_spectrum = offset_spectrum + min(offset_spectrum)
    intp_spectrum = interpolate_spectrum(offset_spectrum, new_x)[0]
    return [intp_spectrum]

def augment_multiplication(spectrum: np.ndarray, new_x: np.ndarray, std: np.ndarray) -> List[np.ndarray]:
    std_spec = std + 1 + np.random.normal(loc=std.mean(), scale=0.2, size=len(std))
    augmented_spectrum = spectrum * std_spec
    intp_spectrum = interpolate_spectrum(augmented_spectrum, new_x)[0]
    return [intp_spectrum]


def prep_data(
    data_df: pd.DataFrame,
    new_x: np.ndarray,
    augmentation: Optional[List[AugmentationCallable]] = None,
    special: str = "N/A",
    mixtures: bool = False
) -> np.ndarray:
    if augmentation is None:
        augmentation = [interpolate_spectrum]

    if mixtures:
        spectra, formula1, formula2, tgt = data_df["spectra"], data_df["formula_1"], data_df["formula_2"], data_df["smiles"]
    else:
        spectra, formula, tgt = data_df["spectra"], data_df["formula"], data_df["smiles"]

    data = list()

    for i, spectrum_orig in enumerate(tqdm.tqdm(spectra)):
        if mixtures:
            formula_temp = split_formula(formula1.iloc[i])
            formula_temp += split_formula(formula2.iloc[i])
        else:
            formula_temp = split_formula(formula.iloc[i])

        tgt_temp = split_smiles(tgt.iloc[i])

        spectra_list = list()

        for aug in augmentation:
            augmented_spectrum = aug(spectrum_orig, new_x)
            spectra_list.extend(augmented_spectrum)

        for spectrum in spectra_list:
            if special == "N/A" or special == "No_Split" or special == "5_Cross" or special == 'Scaffold':
                data.append([tgt_temp, formula_temp + " ".join(spectrum.astype(str))])
            elif special == "Formula":
                data.append([tgt_temp, formula_temp])
            elif special == "Spectrum":
                data.append([tgt_temp, " ".join(spectrum.astype(str))])

    return np.array(data)


def split_train_test_val_random(
    data: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_set, test_set = train_test_split(data, test_size=test_size, random_state=3543)
    train_set, val_set = train_test_split(
        train_set, test_size=val_size, random_state=3543
    )

    return train_set, test_set, val_set


def generate_scaffold(smiles, include_chirality=False):
    """return scaffold string of target molecule"""
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def split_train_test_val_scaffold(data: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    smiles_list = data.smiles

    rng = np.random.RandomState(3543)

    scaffolds = dict()
    for ind, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, True)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = list()
        scaffolds[scaffold].append(ind)

    rng.shuffle(list(scaffolds.values()))

    n_total_test = int(np.floor(test_size * len(smiles_list)))
    n_total_val = int(np.floor(val_size * len(smiles_list)))

    val_index = []
    train_index = []
    test_index = []

    for scaffold_set in scaffolds.values():
        if len(val_index) + len(scaffold_set) <= n_total_val:
                val_index.extend(scaffold_set)
        elif len(test_index) + len(scaffold_set) <= n_total_test:
            test_index.extend(scaffold_set)
        else:
            train_index.extend(scaffold_set)

    return data.iloc[train_index], data.iloc[test_index], data.iloc[val_index]


def prep_data_pipeline(
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    val_set: pd.DataFrame,
    window_vals: np.ndarray,
    augmentation: str,
    special: str,
    out_save_path: str,
    mixtures: bool
) -> None:
    train_data = prep_data(
        train_set,
        window_vals,
        augmentation=augmentation_options[augmentation],
        special=special,
        mixtures = mixtures
    )
    
    save_data_split(train_data, 'train', out_save_path)
    del(train_data)

    test_data = prep_data(test_set, window_vals, special=special, mixtures = mixtures)
    save_data_split(test_data, 'test', out_save_path)

    val_data = prep_data(val_set, window_vals, special=special, mixtures = mixtures)
    save_data_split(val_data, 'val', out_save_path)


def save_data(data: np.ndarray, path: str):
    with open(os.path.join(path, "src-data.txt"), "w") as f:
        for item in data[:, 1]:
            f.write(f"{item}\n")

    with open(os.path.join(path, "tgt-data.txt"), "w") as f:
        for item in data[:, 0]:
            f.write(f"{item}\n")


def save_data_split(
    data: np.ndarray, split: str, path: str
):
    with open(os.path.join(path, f"src-{split}.txt"), "w") as f:
        for item in data[:, 1]:
            f.write(f"{item}\n")

    with open(os.path.join(path, f"tgt-{split}.txt"), "w") as f:
        for item in data[:, 0]:
            f.write(f"{item}\n")


augmentation_options: Dict[str, List[AugmentationCallable]] = {
    "N/A": [interpolate_spectrum],
    "smooth": [interpolate_spectrum, augment_smooth],
    "shift_horizontal": [interpolate_spectrum, augment_shift_horizontal],
    "noise_vertical": [interpolate_spectrum, augment_noise],
    "offset": [interpolate_spectrum, augment__offset],
    "all": [interpolate_spectrum, augment_smooth, augment_shift_horizontal],
}


@click.command()
@click.option("--data_path", required=True, help="Data path")
@click.option("--output_path", required=True, help="Output folder")
@click.option(
    "--n_tokens", default=400, help="Number of tokens to represent the IR spectrum with"
)
@click.option(
    "--window",
    default="full",
    type=click.Choice(["full", "fingerprint", "umir", "merged", "remove_1800"]),
    help="What section of the IR spectrum to use",
)
@click.option(
    "--augmentation",
    default="N/A",
    type=click.Choice(["N/A", "smooth", "shift_horizontal", "noise_vertical", "emsa", "offset", "multiplication", "all"]),
    help="Data augmentation techniques",
)
@click.option(
    "--special",
    default="N/A",
    type=click.Choice(["N/A", "Formula", "Spectrum", "No_Split", "5_Cross", "Scaffold"]),
)
@click.option("--start", default="N/A", type=click.Choice(["N/A", "450", "550", "800"]))
@click.option("--split", default="85_10_5", type=click.Choice(["85_10_5", "70_20_10"]))
@click.option("--mixtures", is_flag=True)

def main(
    data_path: str,
    output_path: str,
    n_tokens: int,
    window: str,
    augmentation: str,
    special: str,
    start: str,
    split: str,
    mixtures: bool
):
    data = load_data(data_path)

    # Deal with EMSA setup
    if augmentation == 'emsa':
        spectra = np.vstack(data.spectra)
        wavenumbers = np.arange(400, 3982, 2)

        reference = spectra.mean(axis=0)
        _, coefs_ = emsc(
            spectra, wavenumbers,
            order=2, reference=reference,
            return_coefs=True)
        coefs_std = coefs_.std(axis=0)

        emsa = EMSA(coefs_std, wavenumbers, reference, order=2)
        augmentation_options['emsa'] = [interpolate_spectrum, partial(augment_emsa, emsa=emsa)]
    elif augmentation == 'multiplication':
        spectra = np.vstack(data.spectra)
        std = np.std(spectra, 0)
        augmentation_options['multiplication'] = [interpolate_spectrum, partial(augment_multiplication, std=std)]

    window_vals = get_window(window, n_tokens, start)

    if special == "No_Split":
        proc_data = prep_data(data, window_vals, special=special, mixtures=mixtures)
        save_data(proc_data, output_path)

    elif special == "5_Cross":
        kf = KFold(n_splits=5)

        for i, (train_index, test_index) in enumerate(kf.split(data)):
            train_set, test_set = data.iloc[train_index], data.iloc[test_index]
            train_set, val_set = train_test_split(train_set, test_size=0.1)

            out_save_path_fold = os.path.join(output_path, f"fold_{i}", "data")
            os.makedirs(out_save_path_fold, exist_ok=True)

            prep_data_pipeline(
                train_set,
                test_set,
                val_set,
                window_vals,
                augmentation,
                special,
                out_save_path_fold,
                mixtures
            )
    elif special == "Scaffold":
        train_set, test_set, val_set = split_train_test_val_scaffold(
                data, test_size=0.2, val_size=0.1
            )
        # Make data directory
        out_save_path = os.path.join(output_path, "data")
        os.makedirs(out_save_path, exist_ok=True)
        
        prep_data_pipeline(
            train_set,
            test_set,
            val_set,
            window_vals,
            augmentation,
            special,
            out_save_path,
            mixtures
        )

    else:
        if split == "85_10_5":
            train_set, test_set, val_set = split_train_test_val_random(
                data, test_size=0.1, val_size=0.05
            )
        elif split == "70_20_10":
            train_set, test_set, val_set = split_train_test_val_random(
                data, test_size=0.2, val_size=0.1
            )
        # Make data directory
        out_save_path = os.path.join(output_path, "data")
        os.makedirs(out_save_path, exist_ok=True)

        prep_data_pipeline(
            train_set,
            test_set,
            val_set,
            window_vals,
            augmentation,
            special,
            out_save_path,
            mixtures
        )


if __name__ == "__main__":
    main()
