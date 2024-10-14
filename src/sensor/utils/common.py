import os
import logging

def create_directories(paths: list):
    """
    Creates directories for the given list of paths.
    :param paths: List of directory paths to create.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logging.info(f"Directory created at {path}")

def get_file_size(file_path: str) -> str:
    """
    Returns the size of the file in bytes.
    :param file_path: Path to the file.
    :return: File size in bytes.
    """
    if os.path.exists(file_path):
        return os.path.getsize(file_path)
    else:
        logging.warning(f"File {file_path} does not exist.")
        return 0
