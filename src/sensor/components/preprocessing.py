import os
import glob
import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler
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
        Load and preprocess .dat files from the given path.
        :param data_path: Path to the raw data.
        :return: Loaded and preprocessed DataFrame.
        """
        dataset_path = os.path.join(data_path, 'Dataset', 'batch*.dat')
        file_paths = glob.glob(dataset_path)
        print(f"Found {len(file_paths)} .dat files to process.")  # Log number of files found

        all_data = []
        
        # Loop through each file path
        for file in file_paths:
            # Load the data using load_svmlight_file
            data, target = load_svmlight_file(file, n_features=128)
            data = data.toarray()  # Convert to dense array
            
            # Group features into 16 sensors (each sensor has 8 features)
            num_sensors = 16
            sensor_data = []
            for sensor_id in range(num_sensors):
                # Select the corresponding 8 features for each sensor
                sensor_features = data[:, sensor_id * 8:(sensor_id + 1) * 8]
                # Aggregate the sensor features (taking the mean)
                sensor_avg = sensor_features.mean(axis=1)
                sensor_data.append(sensor_avg)
            
            # Combine all sensor data into a DataFrame
            sensor_data = pd.DataFrame(sensor_data).T
            sensor_data.columns = [f'sensor_{i+1}' for i in range(num_sensors)]
            # Add the target column (gas class)
            sensor_data['target'] = target
            all_data.append(sensor_data)

        # Concatenate all batch DataFrames into one
        final_df = pd.concat(all_data, ignore_index=True)
        return final_df

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scale the sensor features and encode the target variable.
        :param data: Raw DataFrame.
        :return: Preprocessed DataFrame.
        """
        # Scale features to range (-1, 1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_features = scaler.fit_transform(data.drop('target', axis=1))
        
        # Create a new DataFrame with scaled features
        scaled_df = pd.DataFrame(scaled_features, columns=[f'sensor_{i+1}' for i in range(16)])
        scaled_df['target'] = data['target'].values

        return scaled_df

    def save_preprocessed_data(self, data: pd.DataFrame):
        """
        Save the preprocessed data to a CSV file.
        :param data: Preprocessed DataFrame.
        """
        preprocessed_dir = self.config['preprocessed_dir']
        preprocessed_file = self.config['preprocessed_file']
        
        # Create directories
        create_directories([preprocessed_dir])
        
        # Save the data
        output_file_path = os.path.join(preprocessed_dir, preprocessed_file)
        data.to_csv(output_file_path, index=False)
        print(f"Preprocessed data saved to: {output_file_path}")
