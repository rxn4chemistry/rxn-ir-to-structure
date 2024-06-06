import datetime
import logging
import os
import subprocess
from pathlib import Path
from sys import platform
from typing import Dict

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .gen_spectrum import gen_spectrum

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def gen_emc_file(
    atom_count: int, field: str, smiles: str, save_dir: Path, emc_template: Path
) -> bool:
    if not save_dir.is_dir():
        raise ValueError(f"{save_dir} not a directory.")

    with emc_template.open("r") as f:
        file = f.read()

    formatted_file = file.format(
        atom_count=atom_count, field=field, path=save_dir, smiles=smiles
    )
    save_path = save_dir / "setup.esh"
    with save_path.open("w") as f:
        f.write(formatted_file)

    return True


def run_emc(run_dir: Path) -> bool:
    cwd = run_dir.cwd()
    os.chdir(str(run_dir))

    # Determine the correct EMC command
    if platform == "linux" or platform == "linux2":
        emc_command = "emc_linux64"
    elif platform == "darwin":
        emc_command = "emc_macos"
    elif platform == "win32":
        emc_command = "emc_win32.exe"

    with Path("emc_setup.log").open("w") as out:
        subprocess.call(["emc_setup.pl", "setup.esh"], stdout=out, stderr=out)

    with Path("emc_build.log").open("w") as out:
        subprocess.call([emc_command, "build.emc"], stdout=out, stderr=out)

    os.chdir(str(cwd))

    return True


def gen_lammps_file(
    params_path: Path, data_path: Path, steps: int, save_dir: Path, lammps_template: Path
) -> bool:
    if not save_dir.is_dir():
        raise ValueError(f"{save_dir} not a directory.")

    with lammps_template.open("r") as f:
        file = f.read()

    formatted_file = file.format(params=params_path, data=data_path, steps=steps)
    save_path = save_dir / "lammps.in"
    with save_path.open("w") as f:
        f.write(formatted_file)

    return True


def _replace_setup_line(line: str, box_dim: float) -> str:
    line_split = line.split(" ")
    line_split[-4] = str(box_dim)
    return " ".join(line_split)


def set_box(setup_data_path: Path, box_dim: float = 15.0) -> bool:
    if not setup_data_path.is_file():
        raise ValueError(f"{setup_data_path} is not a file.")

    with setup_data_path.open("r") as f:
        setup_data = f.readlines()

    for i in range(len(setup_data)):
        if "xlo xhi" in setup_data[i]:
            setup_data[i] = _replace_setup_line(setup_data[i], box_dim)
            setup_data[i + 1] = _replace_setup_line(setup_data[i + 1], box_dim)
            setup_data[i + 2] = _replace_setup_line(setup_data[i + 2], box_dim)
            break

    with setup_data_path.open("w") as f:
        f.writelines(setup_data)

    return True


def run_lammps(run_dir: Path, cores: int) -> bool:
    cwd = Path.cwd()
    os.chdir(str(run_dir))

    with Path("lammps_out.log").open("w") as out:
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
    direc: Path,
    steps: int,
    cores: int,
    field: str,
    smiles: str,
    emc_template: Path,
    lammps_template: Path,
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
        if "setup.esh" not in os.listdir(str(direc)):
            raise KeyError("gen_emc Failed, setup.esh not found")

        status["gen_emc_file"] = True
    except KeyError:
        return status

    try:
        run_emc(direc)
        if "setup.data" not in os.listdir(str(direc)) or "setup.params" not in os.listdir(str(direc)):
            raise KeyError("run_emc Failed, setup.data or/and setup.params not found")

        status["run_emc"] = True
    except KeyError:
        return status

    try:
        setup_data_path = direc / "setup.data"
        set_box(setup_data_path)
        status["set_box"] = True
    except IOError:
        return status

    # Run lammps
    try:
        gen_lammps_file(Path("setup.params"), Path("setup.data"), steps, direc, lammps_template)
        if "lammps.in" not in os.listdir(str(direc)):
            raise KeyError("gen_lammps_file Failed, lammps.in not found")

        status["gen_lammps_file"] = True

    except KeyError:
        return status

    try:
        run_lammps(direc, cores)
        if "dipole.txt" not in os.listdir(str(direc)):
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
