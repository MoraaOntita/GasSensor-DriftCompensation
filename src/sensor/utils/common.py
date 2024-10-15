import os
import logging
from sensor.config.configuration import Configuration

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
    
    
def create_required_directories():
    """
    Create necessary directories based on configuration.
    This includes directories for data ingestion, preprocessing, and model saving.
    """
    config = Configuration()  # Instantiate Configuration
    data_ingestion_config = config.get_data_ingestion_config()
    data_preprocessing_config = config.get_data_preprocessing_config()
    model_config = config.get_model_config()

    # List of directories to create
    paths_to_create = [
        data_ingestion_config.get('root_dir', ''),
        data_preprocessing_config.get('preprocessed_dir', ''),
        model_config.get('save_dir', '')
    ]
    
    # Filter out empty paths
    paths_to_create = [path for path in paths_to_create if path]

    # Create directories
    create_directories(paths_to_create)
