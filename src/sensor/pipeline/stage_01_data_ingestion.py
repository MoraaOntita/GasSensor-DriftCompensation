import logging
from sensor.config.configuration import Configuration
from sensor.components.data_ingestion import DataIngestion
from sensor.entity.config_entity import DataIngestionConfig


STAGE_NAME = "Data Ingestion Stage"

class DataIngestionPipeline:
    def __init__(self):
        """
        Initializes the pipeline with configuration and data ingestion components.
        """
        self.config = Configuration()
        self.data_ingestion_config = DataIngestionConfig(**self.config.get_data_ingestion_config())
    
    def main(self):
        """
        Main function that runs the entire data ingestion pipeline.
        """
        data_ingestion = DataIngestion(self.data_ingestion_config)
        data_ingestion.initiate_data_ingestion()


if __name__ == '__main__':
    try:
        # Logging the start of the stage
        logging.info(f"*******************")
        logging.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        
        # Create and run the pipeline
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.main()

        # Logging the successful completion of the stage
        logging.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    
    except Exception as e:
        logging.exception(e)
        raise e
