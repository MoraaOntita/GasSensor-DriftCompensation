import os
import glob
import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler
from sensor.config.configuration import Configuration
from sensor.entity.config_entity import DataPreprocessingConfig
from sensor.utils.common import create_directories
import logging

class Preprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        """
        Initialize preprocessing with config for saving paths.
        
        :param config: DataPreprocessingConfig with paths for preprocessed data.
        """
        self.config = config
        logging.info("Preprocessing initialized with configuration.")

    def run(self, data_path: str):
        """
        Run the entire preprocessing pipeline.
        
        :param data_path: Path to the raw data.
        """
        try:
            # Load raw data
            raw_data = self.load_data(data_path)
            # Preprocess the data for classification
            preprocessed_data = self.preprocess_data(raw_data)
            # Save the preprocessed data
            self.save_preprocessed_data(preprocessed_data)
            logging.info("Preprocessing completed successfully.")
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and preprocess .dat files from the given path.
        
        :param data_path: Path to the raw data.
        :return: Loaded and preprocessed DataFrame.
        """
        try:
            dataset_path = os.path.join(data_path, 'Dataset', 'batch*.dat')
            file_paths = glob.glob(dataset_path)
            logging.info(f"Found {len(file_paths)} .dat files to process.")

            all_data = []

            for file in file_paths:
                sensor_data = self._load_single_file(file)
                all_data.append(sensor_data)

            # Concatenate all batch DataFrames into one
            final_df = pd.concat(all_data, ignore_index=True)
            logging.info("All data loaded successfully.")
            return final_df
        except Exception as e:
            logging.error(f"Error in load_data: {e}")
            raise

    def _load_single_file(self, file_path: str) -> pd.DataFrame:
        """
        Load and preprocess a single .dat file, keeping all features.
        
        :param file_path: Path to the raw .dat file.
        :return: Preprocessed DataFrame for a single file.
        """
        try:
            data, target = load_svmlight_file(file_path, n_features=self.config.num_features)  # Use config for features
            data = data.toarray()

            # Create a DataFrame for the loaded data
            sensor_data = pd.DataFrame(data, columns=[f'feature_{i+1}' for i in range(self.config.num_features)])

            # Add the target column (gas class)
            sensor_data['target'] = target.astype(int)  # Ensure target is an integer
            logging.info(f"Loaded data from {file_path} successfully.")
            return sensor_data
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {e}")
            raise

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data by normalizing the features for classification.
        
        :param data: Raw DataFrame.
        :return: Preprocessed DataFrame.
        """
        try:
            # Normalize the sensor features
            normalized_data = self._normalize_data(data)
            logging.info("Data preprocessing completed successfully.")
            return normalized_data
        except Exception as e:
            logging.error(f"Error during data preprocessing: {e}")
            raise

    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the sensor features to a specified range.
        
        :param data: DataFrame with sensor readings.
        :return: Normalized DataFrame.
        """
        try:
            feature_columns = [col for col in data.columns if 'feature' in col]

            # Scale features to the specified range
            scaler = MinMaxScaler(feature_range=self.config.feature_range)  # Use config for feature range
            scaled_features = scaler.fit_transform(data[feature_columns])

            # Create a new DataFrame with scaled features
            scaled_df = pd.DataFrame(scaled_features, columns=feature_columns)

            # Preserve target column
            scaled_df['target'] = data['target'].values

            logging.info("Normalization completed successfully.")
            return scaled_df
        except Exception as e:
            logging.error(f"Error in normalization: {e}")
            raise

    def save_preprocessed_data(self, data: pd.DataFrame):
        """
        Save the preprocessed data to a CSV file.
        
        :param data: Preprocessed DataFrame.
        """
        try:
            preprocessed_dir = self.config.preprocessed_dir
            preprocessed_file = self.config.preprocessed_file
            
            create_directories([preprocessed_dir])
            
            output_file_path = os.path.join(preprocessed_dir, preprocessed_file)
            data.to_csv(output_file_path, index=False)
            logging.info(f"Preprocessed data saved to: {output_file_path}")
        except Exception as e:
            logging.error(f"Error saving preprocessed data: {e}")
            raise
