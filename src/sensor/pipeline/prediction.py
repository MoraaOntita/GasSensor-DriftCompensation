import os
import logging
import pandas as pd
import tensorflow as tf
from src.sensor.components.data_ingestion import DataIngestion
from src.sensor.components.preprocessing import Preprocessing
from sensor.config.configuration import Configuration
from sensor.entity.config_entity import DataIngestionConfig, DataPreprocessingConfig

class PredictionPipeline:
    def __init__(self, source_url):
        self.source_url = source_url
        self.config = Configuration()

    def run_pipeline(self):
        try:
            # Step 1: Data Ingestion (Download and Extract)
            # Generate the DataIngestionConfig dynamically without modifying the class
            ingestion_config = DataIngestionConfig(
                root_dir="artifacts/data_ingestion",
                source_URL=self.source_url,
                local_data_file="artifacts/data_ingestion/data.zip",
                unzip_dir="artifacts/data_ingestion/extracted"
            )
            data_ingestion = DataIngestion(ingestion_config)
            data_ingestion.initiate_data_ingestion()

            # Step 2: Data Preprocessing
            # Use the config to control the paths dynamically
            preprocessing_config = DataPreprocessingConfig(
                num_features=128,
                feature_range=(0, 1),
                preprocessed_dir="artifacts/preprocessed",
                preprocessed_file="preprocessed_data.csv"
            )
            preprocessing = Preprocessing(preprocessing_config)
            raw_data_dir = ingestion_config.unzip_dir  # Path to the extracted data
            preprocessing.run(raw_data_dir)

            # Step 3: Prediction
            preprocessed_file = os.path.join(preprocessing_config.preprocessed_dir, preprocessing_config.preprocessed_file)
            preprocessed_data = pd.read_csv(preprocessed_file)

            # Load the trained model
            model_path = "artifacts/training/gas_classification_model_final.keras"
            model = tf.keras.models.load_model(model_path)

            # Extract features for prediction
            feature_columns = [col for col in preprocessed_data.columns if 'feature' in col]
            predictions = model.predict(preprocessed_data[feature_columns])

            # Add predictions to the DataFrame
            preprocessed_data['prediction'] = predictions.argmax(axis=1) + 1  # Convert to class labels 1-6

            # Step 4: Save the predictions to CSV
            prediction_output = os.path.join(preprocessing_config.preprocessed_dir, "predicted_output.csv")
            preprocessed_data.to_csv(prediction_output, index=False)

            logging.info(f"Predictions saved to {prediction_output}")
            return prediction_output

        except Exception as e:
            logging.error(f"Error in prediction pipeline: {e}")
            raise
