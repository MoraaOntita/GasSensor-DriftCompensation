import os
import gdown
import zipfile
import logging
from sensor.utils.common import create_directories
from sensor.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def download_data(self):
        """
        Downloads the data from the source URL and saves it locally.
        """
        create_directories([self.data_ingestion_config.root_dir])
        logging.info(f"Downloading dataset from {self.data_ingestion_config.source_URL}")
        gdown.download(self.data_ingestion_config.source_URL, self.data_ingestion_config.local_data_file, quiet=False)
        logging.info(f"Dataset downloaded and saved at {self.data_ingestion_config.local_data_file}")

    def extract_data(self):
        """
        Extracts the downloaded zip file to the specified directory.
        """
        logging.info(f"Extracting {self.data_ingestion_config.local_data_file} to {self.data_ingestion_config.unzip_dir}")
        with zipfile.ZipFile(self.data_ingestion_config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.data_ingestion_config.unzip_dir)
        logging.info(f"Data extracted to {self.data_ingestion_config.unzip_dir}")

    def initiate_data_ingestion(self):
        """
        Initiates the complete data ingestion process: downloading and extraction.
        """
        self.download_data()
        self.extract_data()
