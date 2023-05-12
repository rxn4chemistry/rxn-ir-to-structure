import os

import click
import numpy as np
import pandas as pd
from mol_dyn.utils import load_smiles_csv
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy import interpolate


def get_data(input_folder: str, smiles_path: str) -> pd.DataFrame:
    smiles_csv = load_smiles_csv(smiles_path)

    if "status_lammps.csv" not in os.listdir(input_folder):
        raise ValueError(
            "Expected status_lammps.csv in folder. Verify that the simulation ran correctly"
        )

    status = pd.read_csv(os.path.join(input_folder, "status_lammps.csv"), index_col=0)
    success = status[status["gen_spectrum"] == True]  # noqa: E712

    subfolders = list(success.index)

    data = dict()
    for subfolder in subfolders:
        subfolder_path = os.path.join(input_folder, str(subfolder))

        # Sanity checking
        if (
            subfolder not in list(smiles_csv["Indices"])
            or not os.path.isdir(subfolder_path)
            or "IR-data.csv" not in os.listdir(subfolder_path)
        ):
            continue

        spectra = pd.read_csv(os.path.join(subfolder_path, "IR-data.csv"))
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
    "--simulation_folder", required=True, help="Folder where simulations were run"
)
@click.option(
    "--smiles_path", required=True, help="Folder containing the smiles and indices"
)
@click.option("--output_folder", required=True, help="Output folder")
def main(simulation_folder: str, smiles_path: str, output_folder: str):
    data = get_data(simulation_folder, smiles_path)
    data.to_pickle(os.path.join(output_folder, "ir.pkl"))


if __name__ == "__main__":
    main()
