import contextlib
import os
import pickle
import shutil
import tarfile
from pathlib import Path

from ..utils.logger import get_logger

SP_RESULTS_FILENAME = "results.csv"
SP_SAMPLES_FILENAME = "samples.csv"
SIMULATION_FILENAME = "_sim.pkl"
BASED_SIMULATION_FILENAME = "based_sim.pkl"
SIMULATION_TARFILENAME = "simulations.tar.gz"


logger = get_logger(__name__)


# 改为debug级别
def redirect_stdout_stderr():
    return contextlib.redirect_stderr(open(os.devnull, "w"))
    # return contextlib.redirect_stdout(
    #     open(os.devnull, "w")
    # ), contextlib.redirect_stderr(open(os.devnull, "w"))


def save_plot_data(file_path: Path, **data) -> None:
    """
    Saves the plot data to a specified file path.

    Parameters:
    file_path (str): The path where the data will be saved.
    data (dict): The data to be saved, with variable names as keys and numpy arrays as values.

    Returns:
    None
    """
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_plot_data(file_path: Path) -> dict:
    """
    Loads plot data from a text file with headers.

    Parameters:
    file_path (Path): The path from where the data will be loaded.

    Returns:
    dict: A dictionary with variable names as keys and numpy arrays as values.
    """
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def prepare_save_dir(save_dir: Path) -> Path:
    """
    Prepares the save directory for new data by handling existing directories.

    This function performs the following steps:
    1. Checks if a directory with the same name but with a ".old" suffix exists.
       If it does, it removes this old directory.
    2. If the specified save directory already exists, it renames it by appending
       a ".old" suffix.
    3. Creates a new directory with the specified save directory name.

    Args:
        save_dir (Path): The path to the directory to be prepared.

    Returns:
        Path: The path to the newly created save directory.
    """
    logger.info(f"Preparing save directory: {save_dir}")
    old_save_dir = save_dir.with_suffix(".old")
    if old_save_dir.exists():
        logger.info(f"Removing old save directory: {old_save_dir}")
        shutil.rmtree(old_save_dir)

    if save_dir.exists():
        logger.info(f"Renaming existing save directory to: {old_save_dir}")
        shutil.move(save_dir, old_save_dir)

    logger.info(f"Creating new save directory: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def tar_simulations(save_dir: Path):
    """
    Compresses all .pkl files in the specified directory into a .tar.gz archive and deletes the original .pkl files.
    If no .pkl files are found, the function will log this and skip compression.

    Args:
        save_dir (Path): The directory containing the .pkl files to be compressed.

    Returns:
        None
    """
    pkl_files = list(save_dir.glob(f"*{SIMULATION_FILENAME}"))
    if not pkl_files:
        logger.info(f"No *{SIMULATION_FILENAME} files found to compress")
        return

    logger.info(f"Compressing simulation results in directory: {save_dir}")
    with tarfile.open(save_dir / SIMULATION_TARFILENAME, "w:gz") as tar:
        for file in pkl_files:
            tar.add(file, arcname=file.name)
            file.unlink()
    logger.info("Compression completed")
    return
