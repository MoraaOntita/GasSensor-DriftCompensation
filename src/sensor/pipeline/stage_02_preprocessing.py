import logging
from sensor.config.configuration import Configuration
from sensor.components.preprocessing import Preprocessing

# Define the stage name for logging
STAGE_NAME = "Data Preprocessing Stage"

class PreprocessingPipeline:
    def __init__(self):
        """
        Initializes the pipeline with configuration and preprocessing components.
        """
        self.config = Configuration()
        self.preprocessing_config = self.config.get_data_preprocessing_config()
    
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
