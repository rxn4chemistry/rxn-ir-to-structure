import datetime
import logging
import os
import subprocess
from sys import platform
from typing import Dict

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .gen_spectrum import gen_spectrum

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def gen_emc_file(
    atom_count: int, field: str, smiles: str, save_dir: str, emc_template: str
) -> bool:
    if not os.path.isdir(save_dir):
        raise ValueError(f"{save_dir} not a directory.")

    with open(emc_template, "r") as f:
        file = f.read()

    formatted_file = file.format(
        atom_count=atom_count, field=field, path=save_dir, smiles=smiles
    )
    save_path = os.path.join(save_dir, "setup.esh")
    with open(save_path, "w") as f:
        f.write(formatted_file)

    return True


def run_emc(run_dir: str) -> bool:
    cwd = os.getcwd()
    os.chdir(run_dir)

    # Determine the correct EMC command
    if platform == "linux" or platform == "linux2":
        emc_command = "emc_linux64"
    elif platform == "darwin":
        emc_command = "emc_macos"
    elif platform == "win32":
        emc_command = "emc_win32.exe"

    with open("emc_setup.log", "w") as out:
        subprocess.call(["emc_setup.pl", "setup.esh"], stdout=out, stderr=out)

    with open("emc_build.log", "w") as out:
        subprocess.call([emc_command, "build.emc"], stdout=out, stderr=out)

    os.chdir(cwd)

    return True


def gen_lammps_file(
    params_path: str, data_path: str, steps: int, save_dir: str, lammps_template: str
) -> bool:
    if not os.path.isdir(save_dir):
        raise ValueError(f"{save_dir} not a directory.")

    with open(lammps_template, "r") as f:
        file = f.read()

    formatted_file = file.format(params=params_path, data=data_path, steps=steps)
    save_path = os.path.join(save_dir, "lammps.in")
    with open(save_path, "w") as f:
        f.write(formatted_file)

    return True


def _replace_setup_line(line: str, box_dim: float) -> str:
    line_split = line.split(" ")
    line_split[-4] = str(box_dim)
    return " ".join(line_split)


def set_box(setup_data_path: str, box_dim: float = 15.0) -> bool:
    if not os.path.isfile(setup_data_path):
        raise ValueError(f"{setup_data_path} not a directory.")

    with open(setup_data_path, "r") as f:
        setup_data = f.readlines()

    for i in range(len(setup_data)):
        if "xlo xhi" in setup_data[i]:
            setup_data[i] = _replace_setup_line(setup_data[i], box_dim)
            setup_data[i + 1] = _replace_setup_line(setup_data[i + 1], box_dim)
            setup_data[i + 2] = _replace_setup_line(setup_data[i + 2], box_dim)
            break

    with open(setup_data_path, "w") as f:
        f.writelines(setup_data)

    return True


def run_lammps(run_dir: str, cores: int) -> bool:
    cwd = os.getcwd()
    os.chdir(run_dir)

    with open("lammps_out.log", "w") as out:
        if cores == 1:
            subprocess.call(
                ["lmp", "-in", "lammps.in"],
                stdout=out,
                stderr=out,
            )
        else:
            subprocess.call(
                ["mpirun", "-np", str(cores), "lmp_mpi", "-in", "lammps.in"],
                stdout=out,
                stderr=out,
            )

    os.chdir(cwd)
    return True


def pipeline(
    direc: str,
    steps: int,
    cores: int,
    field: str,
    smiles: str,
    emc_template: str,
    lammps_template: str,
) -> Dict[str, bool]:
    status = {
        "gen_emc_file": False,
        "run_emc": False,
        "set_box": False,
        "gen_lammps_file": False,
        "run_lammps": False,
        "gen_spectrum": False,
    }

    t1 = datetime.datetime.now()

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError()
        atom_count = rdMolDescriptors.CalcNumAtoms(mol)
    except ValueError:
        return status

    try:
        gen_emc_file(atom_count, field, smiles, direc, emc_template)
        if "setup.esh" not in os.listdir(direc):
            raise KeyError("gen_emc Failed, setup.esh not found")

        status["gen_emc_file"] = True
    except KeyError:
        return status

    try:
        run_emc(direc)
        if "setup.data" not in os.listdir(direc) or "setup.params" not in os.listdir(
            direc
        ):
            raise KeyError("run_emc Failed, setup.data or/and setup.params not found")

        status["run_emc"] = True
    except KeyError:
        return status

    try:
        setup_data_path = os.path.join(direc, "setup.data")
        set_box(setup_data_path)
        status["set_box"] = True
    except IOError:
        return status

    # Run lammps
    try:
        gen_lammps_file("setup.params", "setup.data", steps, direc, lammps_template)
        if "lammps.in" not in os.listdir(direc):
            raise KeyError("gen_lammps_file Failed, lammps.in not found")

        status["gen_lammps_file"] = True

    except KeyError:
        return status

    try:
        run_lammps(direc, cores)
        if "dipole.txt" not in os.listdir(direc):
            raise KeyError("run_lammps Failed, dipole.txt not found")

        status["run_lammps"] = True
    except KeyError:
        return status

    # Generate spectrum
    try:
        gen_spectrum(direc)
        if "IR-data.csv" not in os.listdir(direc):
            raise KeyError("gen_spectrum Failed, IR-data.csv not found")

        status["gen_spectrum"] = True
    except KeyError:
        return status

    t2 = datetime.datetime.now()
    logger.info(f"Steps: {steps} Time: {t2 - t1}")
    return status
