import os
import glob
import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
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
            raw_data = self.load_data(data_path)
            preprocessed_data = self.preprocess_data(raw_data)
            self.save_preprocessed_data(preprocessed_data)
            logging.info("Preprocessing completed successfully.")
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and concatenate all .dat files into a single DataFrame.
        
        :param data_path: Path to the raw data.
        :return: Concatenated DataFrame.
        """
        try:
            dataset_path = os.path.join(data_path, 'Dataset', 'batch*.dat')
            file_paths = glob.glob(dataset_path)
            logging.info(f"Found {len(file_paths)} .dat files to process.")

            all_data = [self._load_single_file(file) for file in file_paths]
            final_df = pd.concat(all_data, ignore_index=True)
            logging.info("All data loaded successfully.")
            return final_df
        except Exception as e:
            logging.error(f"Error in load_data: {e}")
            raise

    def _load_single_file(self, file_path: str) -> pd.DataFrame:
        """
        Load and format a single .dat file.
        
        :param file_path: File path.
        :return: DataFrame with features and target.
        """
        try:
            data, target = load_svmlight_file(file_path, n_features=self.config.num_features)
            df = pd.DataFrame(data.toarray(), columns=[f'feature_{i+1}' for i in range(self.config.num_features)])
            df['target'] = target.astype(int)
            logging.info(f"Loaded data from {file_path}.")
            return df
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {e}")
            raise

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply skewness correction and normalization.
        
        :param data: Raw data.
        :return: Preprocessed data.
        """
        try:
            feature_columns = [col for col in data.columns if 'feature' in col]
            data_copy = data.copy()

            # Skewness detection
            skewness = data_copy[feature_columns].skew()
            high_skewed = skewness[abs(skewness) > 1]

            if not high_skewed.empty:
                logging.info(f"Applying skewness correction on: {list(high_skewed.index)}")
                data_copy = self._transform_skewed_features(data_copy, high_skewed)

            # Normalization
            data_copy = self._normalize_data(data_copy)

            return data_copy
        except Exception as e:
            logging.error(f"Error in preprocess_data: {e}")
            raise

    def _transform_skewed_features(self, df: pd.DataFrame, skewed_features: pd.Series) -> pd.DataFrame:
        try:
            df_copy = df.copy()
            pt = PowerTransformer(method='yeo-johnson')

            for feature in skewed_features.index:
                try:
                    if skewed_features[feature] > 1:
                        df_copy[feature] = np.log1p(df_copy[feature])
                    df_copy[[feature]] = pt.fit_transform(df_copy[[feature]])
                except Exception as e:
                    logging.warning(f"Skipping transformation for {feature}: {e}")

            return df_copy
        except Exception as e:
            logging.warning(f"Error transforming skewed features: {e}")
            return df

    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply MinMax scaling.
        
        :param data: Input DataFrame.
        :return: Scaled DataFrame.
        """
        try:
            feature_columns = [col for col in data.columns if 'feature' in col]
            scaler = MinMaxScaler(feature_range=self.config.feature_range)
            scaled_features = scaler.fit_transform(data[feature_columns])

            scaled_df = pd.DataFrame(scaled_features, columns=feature_columns)
            scaled_df['target'] = data['target'].values

            logging.info("Feature normalization completed.")
            return scaled_df
        except Exception as e:
            logging.error(f"Error in normalization: {e}")
            raise

    def save_preprocessed_data(self, data: pd.DataFrame):
        """
        Save the final preprocessed data to CSV.
        
        :param data: Preprocessed data.
        """
        try:
            preprocessed_dir = self.config.preprocessed_dir
            preprocessed_file = self.config.preprocessed_file

            create_directories([preprocessed_dir])
            output_file_path = os.path.join(preprocessed_dir, preprocessed_file)

            data.to_csv(output_file_path, index=False)
            logging.info(f"Preprocessed data saved to {output_file_path}")
        except Exception as e:
            logging.error(f"Error saving preprocessed data: {e}")
            raise
