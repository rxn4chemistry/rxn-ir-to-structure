import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy import interpolate

from ir_to_struc.mol_dyn.utils import load_smiles_csv


def get_data(input_folder: Path, smiles_path: Path) -> pd.DataFrame:
    smiles_csv = load_smiles_csv(smiles_path)

    if "status_lammps.csv" not in os.listdir(input_folder):
        raise ValueError(
            "Expected status_lammps.csv in folder. Verify that the simulation ran correctly"
        )

    status = pd.read_csv(input_folder / "status_lammps.csv", index_col=0)
    success = status[status["gen_spectrum"] == True]  # noqa: E712

    subfolders = list(success.index)

    data = dict()
    for subfolder in subfolders:
        subfolder_path = input_folder / str(subfolder)

        # Sanity checking
        if (
            subfolder not in list(smiles_csv["Indices"])
            or not subfolder_path.is_dir()
            or "IR-data.csv" not in os.listdir(subfolder_path)
        ):
            continue

        spectra = pd.read_csv(subfolder_path / "IR-data.csv")
        spectra_relevant = spectra.iloc[198:1991]

        spectra_relevant_np = spectra_relevant[
            ["# Frequency(cm^-1)", " Spectra_qm"]
        ].to_numpy()

        try:
            interpolation = interpolate.interp1d(
                spectra_relevant_np[:, 0], spectra_relevant_np[:, 1]
            )

            wave_nums = np.arange(400, 3982, 2)
            intensities = interpolation(wave_nums)
            intensities_normalised = intensities / sum(intensities)
        except ValueError:
            continue

        try:
            smiles = smiles_csv[smiles_csv["Indices"] == subfolder].Smiles.item()

            mol = Chem.MolFromSmiles(smiles)
            formula = rdMolDescriptors.CalcMolFormula(mol)
            smiles = Chem.MolToSmiles(mol)
        except TypeError:
            continue

        data[subfolder] = {
            "smiles": smiles,
            "formula": formula,
            "spectra": intensities_normalised,
        }

    return pd.DataFrame.from_dict(data, orient="index")


@click.command()
@click.option(
    "--simulation_folder", type=Path, required=True, help="Folder where simulations were run"
)
@click.option(
    "--smiles_path", type=Path, required=True, help="Folder containing the smiles and indices"
)
@click.option("--output_folder", type=Path, required=True, help="Output folder")
def main(simulation_folder: Path, smiles_path: Path, output_folder: Path):
    data = get_data(simulation_folder, smiles_path)
    data.to_pickle(output_folder / "ir.pkl")


if __name__ == "__main__":
    main()
