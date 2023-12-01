import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np

_logger = logging.getLogger("xtb")


def xtb_calculate(
    atoms: List[str],
    coords: List[list],
    charge: int = 0,
    multiplicity: int = 1,
    options: dict = {},
    scr: str = ".",
    n_cores: int = 1,
    detailed_input: dict = None,
    xtb_cmd: str = "xtb",
) -> dict:
    """Run xTB calculation.

    Args:
        atoms (List[str]): List of atom symbols.
        coords (List[list]): 3xN list of atom coordinates.
        charge (int): Formal charge of molecule.
        multiplicity (int): Spin multiplicity of molecule.
        options (dict): xTB calculation options.
        scr (str, optional): Path to scratch directory. Defaults to '.'.
        n_cores (int, optional): Number of cores used in calculation. Defaults to 1.
        detailed_input (dict, optional): Detailed input for xTB calculation. Defaults to None.
        xtb_cmd (str): Path to xtb executable.

    Returns:
        dict: {'atoms': ..., 'coords': ..., ...}
    """
    set_threads(n_cores)

    # create TMP directory
    tempdir = tempfile.TemporaryDirectory(dir=scr, prefix="XTB_")
    tmp_scr = Path(tempdir.name)
    xyz_file = write_xyz(atoms, coords, tmp_scr)

    # clean xtb method option
    for k, value in options.items():
        if "gfn" in k.lower():
            if value is not None and value is not True:
                options[k + str(value)] = None
                del options[k]
                break

    # options to xTB command
    cmd = f"{xtb_cmd} --chrg {charge} --uhf {multiplicity-1} --norestart --verbose --parallel {n_cores} "
    for key, value in options.items():
        if value is None or value is True:
            cmd += f"--{key} "
        else:
            cmd += f"--{key} {str(value)} "
    if detailed_input is not None:
        fpath = write_detailed_input(detailed_input, tmp_scr)
        cmd += f"--input {fpath.name} "

    lines = run_xtb((cmd, xyz_file))
    if not normal_termination(lines):
        _logger.warning("xTB did not terminate normally")
        _logger.info("".join(lines))
        results = {"normal_termination": False, "log": "".join(lines)}
        return results

    # read results
    results = read_xtb_results(lines)
    if "hess" in options:
        results.update(read_thermodynamics(lines))
    if "grad" in options:
        with open(tmp_scr / "mol.engrad", "r") as f:
            grad_lines = f.readlines()
        results["grad"] = read_gradients(grad_lines)
    if "wbo" in options:
        results["wbo"] = read_wbo(tmp_scr / "wbo", len(atoms))
    if "pop" in options:
        results["mulliken"] = read_mulliken(tmp_scr / "charges")
    results["atoms"] = atoms
    results["coords"] = coords
    if "opt" in options:
        results["opt_coords"] = read_opt_structure(lines)[-1]

    return results


def set_threads(n_cores: int):
    """Set threads and procs environment variables."""
    _ = list(stream("ulimit -s unlimited"))
    os.environ["OMP_STACKSIZE"] = "4G"
    os.environ["OMP_NUM_THREADS"] = f"{n_cores},1"
    os.environ["MKL_NUM_THREADS"] = str(n_cores)
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

    _logger.debug("Set OMP_STACKSIZE to 4G")
    _logger.debug(f"Set OMP_NUM_THREADS to {n_cores},1")
    _logger.debug(f"Set MKL_NUM_THREADS to {n_cores}")
    _logger.debug("Set OMP_MAX_ACTIVE_LEVELS to 1")


def write_xyz(atoms: List[str], coords: List[list], scr: Path):
    """Write xyz coordinate file."""
    natoms = len(atoms)
    xyz = f"{natoms} \n \n"
    for atomtype, coord in zip(atoms, coords):
        xyz += f"{atomtype}  {' '.join(list(map(str, coord)))} \n"
    with open(scr / "mol.xyz", "w") as inp:
        inp.write(xyz)
    _logger.debug(f"Written xyz-file to {scr / 'mol.xyz'}")
    return scr / "mol.xyz"


def write_detailed_input(details_dict: dict, scr: Path):
    """Write detailed input file for xTB calculation."""
    detailed_input_str = ""
    for key, value in details_dict.items():
        detailed_input_str += f"${key}\n"
        for subkey, subvalue in value.items():
            detailed_input_str += (
                f'\t{subkey}: {", ".join([str(i) for i in subvalue])}\n'
            )
    detailed_input_str += "$end\n"

    fpath = scr / "details.inp"
    _logger.debug(f"Writing detailed input to {fpath}")
    _logger.debug(detailed_input_str)
    with open(fpath, "w") as inp:
        inp.write(detailed_input_str)

    return fpath


def run_xtb(args: Tuple[str]):
    """Run xTB command for xyz-file in parent directory, logs and returns
    output."""
    cmd, xyz_file = args
    generator = stream(f"{cmd}-- {xyz_file.name}", cwd=xyz_file.parent)
    lines = []
    for line in generator:
        lines.append(line)
        _logger.debug(line.rstrip("\n"))
    return lines


def normal_termination(lines: List[str]):
    """Check if xTB terminated normally."""
    for line in reversed(lines):
        if line.strip().startswith("normal termination"):
            return True
    return False


def read_opt_structure(lines: List[str]):
    """Read optimized structure from xTB output."""

    def _parse_coordline(line: str):
        """Parse coordinate line from xyz-file."""
        line = line.split()
        atom = line[0]
        coord = [float(x) for x in line[-3:]]
        return atom, coord

    for i, l in reversed(list(enumerate(lines))):
        if "final structure" in l:
            break
    n_atoms = int(lines[i + 2].rstrip())
    start = i + 4
    end = start + n_atoms
    atoms = []
    coords = []
    for line in lines[start:end]:
        atom, coord = _parse_coordline(line)
        atoms.append(atom)
        coords.append(coord)

    return atoms, coords


def _sanitize_name(prop_name: str):
        return prop_name.replace('_', '-').replace(' ', '_').replace('.', '').strip()

def read_thermodynamics(lines: List[str]):
    """Read thermodynamics output of frequency calculation."""

    thermo_idx = np.nan
    thermo_properties = {}
    for i, line in enumerate(lines):
        line = line.strip()
        if "THERMODYNAMIC" in line:
            thermo_idx = i
        if i > (thermo_idx + 2):
            if 20 * ":" in line:
                thermo_idx = np.nan
            elif 20 * "." in line:
                continue
            else:
                tmp = line.strip(":").strip().strip("->").split()[:-1]
                thermo_properties[_sanitize_name(" ".join(tmp[:-1]))] = float(tmp[-1])
    return thermo_properties


def read_gradients(lines: List[str]):
    """Read gradients from engrad file."""
    gradients = []
    gradient_idx = np.nan
    for i, line in enumerate(lines):
        line = line.strip()
        if "gradient" in line:
            gradient_idx = i
        if i > (gradient_idx + 1):
            if line.startswith("#"):
                gradient_idx = np.nan
                break
            gradients.append(float(line))
    gradients = np.asarray(gradients)
    return gradients.reshape(-1, 3)


def read_wbo(wbo_file, n_atoms):
    """Read Wiberg bond order from wbo file."""
    data = np.loadtxt(wbo_file)
    wbo = np.zeros((n_atoms, n_atoms))
    for i, j, o in data:
        i = i.astype(int) - 1
        j = j.astype(int) - 1
        wbo[i, j] = wbo[j, i] = o
    return wbo


def read_mulliken(charges_file):
    """Read Mulliken charges from charges file."""
    return np.loadtxt(charges_file)


def read_xtb_results(lines: List[str]):
    """Read basic results from xTB log."""

    def _get_runtime(lines: List[str]):
        """Reads xTB runtime in seconds."""
        _, _, days, _, hours, _, minutes, _, seconds, _ = line.strip().split()
        total_seconds = (
            float(seconds)
            + 60 * float(minutes)
            + 360 * float(hours)
            + 86400 * float(days)
        )
        return total_seconds

    property_start_idx, dipole_idx, quadrupole_idx, runtime_idx, polarizability_idx = (
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    )
    properties = {}
    for i, line in enumerate(lines):
        line = line.strip()
        if "xtb version" in line:
            xtb_version = line.split()[3]
        elif "program call" in line:
            programm_call = line.split(":")[-1]
        elif "SUMMARY" in line:
            property_start_idx = i
        elif "molecular dipole" in line:
            dipole_idx = i
        elif "molecular quadrupole" in line:
            quadrupole_idx = i
        elif "total:" in line:
            runtime_idx = i
        elif "Mol. α(0) /au" in line:
            polarizability_idx = i

        # read property table
        if i > (property_start_idx + 1):
            if 20 * ":" in line:
                property_start_idx = np.nan
            elif 20 * "." in line:
                continue
            else:
                tmp = line.strip(":").strip().strip("->").strip('.').split()[:-1]
                properties[_sanitize_name(" ".join(tmp[:-1]))] = float(tmp[-1])

        # read polarizability
        if i == polarizability_idx:
            polarizability = float(line.split()[-1])

        # read dipole moment
        if i > (dipole_idx + 2):
            dip_x, dip_y, dip_z, dip_norm = [
                float(x) for x in line.split()[1:]
            ]  # norm is in Debye
            dipole_vec = np.array(
                [dip_x, dip_y, dip_z]
            )  # in a.u. (*2.5412 ot convert to Debye)
            dipole_idx = np.nan

        # read quadrupole moment
        if i > (quadrupole_idx + 3):
            quad_vec = np.array([float(x) for x in line.split()[1:]])  # in a.u.
            quad_vec[2], quad_vec[3] = quad_vec[3], quad_vec[2]
            quadrupole_mat = np.zeros((3, 3))
            indices = np.triu_indices(3)
            quadrupole_mat[indices] = quad_vec
            quadrupole_mat[indices[::-1]] = quad_vec
            quadrupole_idx = np.nan

        # read runtimes
        if i > runtime_idx:
            if i == (runtime_idx + 1):
                wall_time = _get_runtime(line)
            else:
                cpu_time = _get_runtime(line)
                runtime_idx = np.nan

    results = {
        "normal_termination": True,
        "programm_call": programm_call,
        "programm_version": xtb_version,
        "wall_time": wall_time,
        "cpu_time": cpu_time,
        "polarizability": polarizability,
        "dipole_vec": dipole_vec,
        "dipole_norm": dip_norm,
        "quadrupole_mat": quadrupole_mat,
    }

    results.update(properties)

    return results


def stream(cmd, cwd=None, shell=True):
    """Execute command in directory, and stream stdout."""
    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=shell,
        cwd=cwd,
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    # Yield errors
    stderr = popen.stderr.read()
    popen.stdout.close()
    yield stderr

    return


if __name__ == "__main__":
    atoms = ["C", "C", "C", "N", "C", "C", "N", "H", "H", "H", "H", "H", "H", "H"]
    coords = [
        [-0.150572945378, 1.06902383757551, 0.13369717980808],
        [-1.53340374624082, 0.97192911929508, 0.16014351260219],
        [-2.11046032356157, -0.28452881345742, 0.02365204850958],
        [-1.40326713049944, -1.39105857658305, -0.1287228595083],
        [-0.08701882046636, -1.30624454138783, -0.15461643853251],
        [0.58404976357399, -0.09554934782587, -0.02805090001267],
        [2.04601334595326, -0.0521981720584, -0.06393160516877],
        [0.3301095007743, 2.02983678857648, 0.23606713936603],
        [-2.14949770349705, 1.84686355411139, 0.28440826742643],
        [-3.18339157653865, -0.41415380049138, 0.03929295998759],
        [0.43604958186272, -2.24346585185004, -0.28257385567194],
        [2.37469662935334, 0.91726637325056, 0.04696293388512],
        [2.40041378960073, -0.42116004539483, -0.95829570995538],
        [2.44627963506354, -0.62656052376019, 0.69196732726456],
    ]

    options = {"opt": True, "alpb": "methanol", "wbo": True, "pop": True}
    results = xtb_calculate(atoms=atoms, coords=coords, charge=1, options=options)

    for key, value in results.items():
        print(key)
        print(value)
        print(80 * "*")
