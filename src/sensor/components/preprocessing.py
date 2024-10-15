import os
import glob
import pandas as pd
import numpy as np
from sensor.config.configuration import Configuration
from sensor.entity.config_entity import DataPreprocessingConfig
from sensor.utils.common import create_directories

class Preprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        """
        Initialize preprocessing with config for saving paths.
        :param config: DataPreprocessingConfig with paths for preprocessed data.
        """
        self.config = config

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load dataset from the given path.
        :param data_path: Path to the raw data.
        :return: Loaded DataFrame.
        """
        # Fetch dataset path and adjust it to the correct folder
        dataset_path = os.path.join(data_path, 'Dataset', 'batch*.dat')

        # Use glob to find all .dat files in the specified directory
        file_paths = glob.glob(dataset_path)
        print(f"Found {len(file_paths)} .dat files to process.")  # Log number of files found

        dataframes = []

        # Loop through each file path
        for file in file_paths:
            with open(file, 'r') as f:
                for line in f:
                    values = line.strip().split(' ')
                    label = values[0]
                    sensor_data = {int(kv.split(':')[0]): float(kv.split(':')[1]) for kv in values[1:]}
                    sensor_data['label'] = label
                    dataframes.append(pd.DataFrame(sensor_data, index=[0]))

        # Check if data was found, raise an error if no files were read
        if not dataframes:
            raise ValueError("No data found to load. Please check the file paths.")
        
        # Concatenate all DataFrames into a single DataFrame
        data = pd.concat(dataframes, ignore_index=True)
        return data

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data by transforming skewed features and encoding target variable.
        :param data: Raw DataFrame.
        :return: Preprocessed DataFrame.
        """
        # Rename columns
        sensor_columns = [f'sensor_{i}' for i in range(1, 129)]
        sensor_columns.append('gas_class')
        data.columns = sensor_columns

        # Handle skewed features
        skewed_features = data[sensor_columns[:-1]].apply(lambda x: x.skew()).sort_values(ascending=False)
        skewness = skewed_features[skewed_features > 0.5]
        for feature in skewness.index:
            data[feature] = np.log1p(data[feature])  # log1p handles zero values

        # One-hot encode the target variable (gas_class)
        data = pd.get_dummies(data, columns=['gas_class'], drop_first=True)

        return data

    def save_preprocessed_data(self, data: pd.DataFrame):
        """
        Save the preprocessed data to a CSV file.
        :param data: Preprocessed DataFrame.
        """
        # Access config values using key-based indexing
        preprocessed_dir = self.config['preprocessed_dir']
        preprocessed_file = self.config['preprocessed_file']
        
        # Create directories
        create_directories([preprocessed_dir])
        
        # Save the data
        output_file_path = os.path.join(preprocessed_dir, preprocessed_file)
        data.to_csv(output_file_path, index=False)
        print(f"Preprocessed data saved to: {output_file_path}")

