import os
import yaml
from sensor.constants import CONFIG_FILE_PATH
import logging

class Configuration:
    def __init__(self, config_file_path=CONFIG_FILE_PATH):
        """
        Initialize the configuration by reading from the YAML file.
        :param config_file_path: Path to the configuration YAML file.
        """
        self.config_file_path = config_file_path
        self.config = self.read_yaml_file()

    def read_yaml_file(self):
        """
        Read the YAML configuration file.
        :return: Parsed YAML file content as a dictionary.
        """
        logging.info(f"Reading configuration file from {self.config_file_path}")
        try:
            with open(self.config_file_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Error reading config file: {e}")
            raise e

    def get_data_ingestion_config(self):
        """
        Get data ingestion related configurations from the YAML file.
        :return: Data ingestion configurations as a dictionary.
        """
        return self.config.get('data_ingestion', {})
    
    
    def get_data_preprocessing_config(self):
        """
        Get data preprocessing related configurations from the YAML file.
        :return: Data preprocessing configurations as a dictionary.
        """
        return self.config.get('data_preprocessing', {})
    

    def get_prepare_base_model_config(self):
        """
        Get base model related configurations from the YAML file.
        :return: Base model configurations as a dictionary.
        """
        return self.config.get('prepare_base_model', {})