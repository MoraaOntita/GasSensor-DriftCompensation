import logging
from sensor.config.configuration import Configuration
from sensor.components.preprocessing import Preprocessing
from sensor.entity.config_entity import DataPreprocessingConfig

# Define the stage name for logging
STAGE_NAME = "Data Preprocessing Stage"

class PreprocessingPipeline:
    def __init__(self):
        """
        Initializes the pipeline with configuration and preprocessing components.
        """
        self.config = Configuration()
        preprocessing_config_dict = self.config.get_data_preprocessing_config()
        
        # Convert the dictionary to DataPreprocessingConfig object
        self.preprocessing_config = DataPreprocessingConfig(
            preprocessed_dir=preprocessing_config_dict['preprocessed_dir'],
            preprocessed_file=preprocessing_config_dict['preprocessed_file'],
            num_features=preprocessing_config_dict['num_features'],
            feature_range=tuple(preprocessing_config_dict['feature_range'])
        )
    
    def main(self):
        """
        Main function that runs the entire preprocessing pipeline.
        """
        preprocessing = Preprocessing(config=self.preprocessing_config)

        # Get data path from config and add the 'Dataset' subdirectory
        data_path = self.config.get_data_ingestion_config().get('unzip_dir', '')
        raw_data = preprocessing.load_data(data_path)

        # Preprocess the data
        preprocessed_data = preprocessing.preprocess_data(raw_data)

        # Save preprocessed data
        preprocessing.save_preprocessed_data(preprocessed_data)
        
        
if __name__ == '__main__':
    try:
        # Logging the start of the stage
        logging.info(f"*******************")
        logging.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        
        # Create and run the pipeline
        preprocessing_pipeline = PreprocessingPipeline()
        preprocessing_pipeline.main()

        # Logging the successful completion of the stage
        logging.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    
    except Exception as e:
        logging.exception(e)
        raise e
