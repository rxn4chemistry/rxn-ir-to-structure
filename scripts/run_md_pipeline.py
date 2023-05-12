import logging
import os

import click
import pandas as pd
import tqdm
from mol_dyn.pipeline import pipeline
from mol_dyn.utils import load_smiles_csv


@click.command()
@click.option("--smiles_csv", required=True, help="Smiles csv")
@click.option("--output_folder", required=True, help="Output folder")
@click.option("--field", type=str, default="pcff", help="Force field")
@click.option("--steps", type=int, default=500000, help="Steps")
@click.option("--cores", type=int, default=8, help="Cores")
@click.option(
    "--emc_template",
    type=str,
    default="mol_dyn/emc_template.esh",
    help="Path to the emc template file in the mol_dyn folder",
)
@click.option(
    "--lammps_template",
    type=str,
    default="mol_dyn/lammps_template.in",
    help="Path to the lammps `template file in the mol_dyn folder",
)
def main(
    smiles_csv: str,
    output_folder: str,
    field: str = "pcff",
    steps: int = 500000,
    cores: int = 8,
    emc_template: str = "mol_dyn/emc_template.esh",
    lammps_template: str = "mol_dyn/lammps_template.in",
):
    logging.basicConfig(level="INFO")

    smiles_data = load_smiles_csv(smiles_csv)
    os.makedirs(output_folder, exist_ok=True)

    status_dict = dict()
    for i in tqdm.tqdm(range(len(smiles_data))):
        # index corresponds to the indice of the row in the smiles csv. The scripts creates a folder for each index, making it easy to correlate the smiles to the simulation folder
        index, smiles = smiles_data.iloc[i]["Indices"], smiles_data.iloc[i]["Smiles"]

        folder_path = os.path.join(output_folder, str(index))
        os.makedirs(folder_path, exist_ok=True)

        status = pipeline(
            folder_path, steps, cores, field, smiles, emc_template, lammps_template
        )

        if not status["gen_emc_file"]:
            logging.warning(
                f"Simulation for index {index} and SMILES {smiles} failed to generate emc input file."
            )
        elif not status["run_emc"]:
            logging.warning(
                f"Simulation for index {index} and SMILES {smiles} failed to run emc."
            )
        elif not status["gen_lammps_file"]:
            logging.warning(
                f"Simulation for index {index} and SMILES {smiles} failed to generate lammps input file."
            )
        elif not status["run_lammps"]:
            logging.warning(
                f"Simulation for index {index} and SMILES {smiles} failed to run lammps."
            )
        elif not status["gen_spectrum"]:
            logging.warning(
                f"Simulation for index {index} and SMILES {smiles} failed to create spectrum from simulation."
            )

        status_dict[index] = status

    status_df = pd.DataFrame.from_dict(status_dict, orient="index")

    logging.info(
        "Success: {} of total {}: {:.3f}%".format(
            status_df["gen_spectrum"].sum(),
            len(status_df),
            status_df["gen_spectrum"].sum() / len(status_df) * 100,
        )
    )
    status_df.to_csv(os.path.join(output_folder, "status_lammps.csv"))


if __name__ == "__main__":
    main()
