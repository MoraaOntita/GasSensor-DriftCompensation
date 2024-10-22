import os
import yaml
import logging
from sensor.constants import CONFIG_FILE_PATH
from sensor.entity.config_entity import ModelConfig
from dotenv import load_dotenv

load_dotenv()


class Configuration:
    def __init__(self, config_file_path='config/config.yaml', param_file_path='param.yaml'):
        """
        Initializes the Configuration class by loading the YAML configuration files.

        :param config_file_path: Path to the main configuration file (default: 'config/config.yaml').
        :param param_file_path: Path to the parameters configuration file (default: 'param.yaml').
        """
        self.config_file_path = config_file_path
        self.param_file_path = param_file_path
        self.config = self.read_yaml_file(self.config_file_path)
        self.params = self.read_yaml_file(self.param_file_path)

    def read_yaml_file(self, file_path):
        """
        Reads a YAML configuration file.

        :param file_path: Path to the YAML file to read.
        :return: Parsed YAML file content as a dictionary.
        """
        logging.info(f"Reading configuration file from {file_path}")
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Error reading config file: {e}")
            raise e

    def get_data_ingestion_config(self):
        """
        Gets data ingestion related configurations from the YAML file.

        :return: Data ingestion configurations as a dictionary.
        """
        return self.config.get('data_ingestion', {})
    
    def get_data_preprocessing_config(self):
        """
        Gets data preprocessing related configurations from the YAML file.

        :return: Data preprocessing configurations as a dictionary.
        """
        return self.config.get('data_preprocessing', {})
    
    def get_model_config(self):
        """
        Gets model configuration from the loaded configuration.
        :return: Dictionary containing model configuration data.
        """
        model_config_data = self.config.get('prepare_base_model', {})
        logging.info(f"Model config data: {model_config_data}, Type: {type(model_config_data)}")
        
        if isinstance(model_config_data, dict):
            # Do not return ModelConfig here; instead, return the dictionary directly
            return model_config_data
        else:
            logging.error("Expected a dictionary for model configuration.")
            raise ValueError("Model configuration data is not valid.")


    def get_training_params(self):
        """
        Gets training parameters from the parameter configuration file.

        :return: Training parameters as a dictionary.
        """
        return self.params.get('model_training', {})
    
        
    def get_training_data_path(self):
        """
        Returns the path to the preprocessed training data file.
        """
        preprocessed_file = self.get_data_preprocessing_config()['preprocessed_file']
        preprocessed_dir = self.get_data_preprocessing_config()['preprocessed_dir']
        return os.path.join(preprocessed_dir, preprocessed_file)

    
    def get_prepare_base_model_config(self):
        """
        Retrieves the base model preparation configuration.

        :return: Dictionary containing the base model preparation configuration.
        """
        try:
            return self.config.get('prepare_base_model', {})
        except KeyError as e:
            logging.error(f"Key 'prepare_base_model' not found in the configuration file: {e}")
            raise e


    def get_mlflow_config(self):
        """
        Retrieves MLflow configuration from environment variables.

        :return: Dictionary containing MLflow configuration.
        """
        return {
            "tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
            "username": os.getenv("MLFLOW_TRACKING_USERNAME"),
            "password": os.getenv("MLFLOW_TRACKING_PASSWORD"),
        }