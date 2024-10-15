import logging
from src.sensor.pipeline.stage_01_data_ingestion import DataIngestionPipeline  # Update this import
from src.sensor.pipeline.stage_02_preprocessing import PreprocessingPipeline


DATA_INGESTION_STAGE_NAME = "Data Ingestion Stage"


def run_data_ingestion():
    """
    Runs the data ingestion stage of the pipeline.
    """
    try:
        logging.info(f"Starting {DATA_INGESTION_STAGE_NAME}.")
        data_ingestion_pipeline = DataIngestionPipeline()  # Create an instance of the pipeline
        data_ingestion_pipeline.main()  # Call the main method to run it
        logging.info(f"{DATA_INGESTION_STAGE_NAME} completed successfully.")
    except Exception as e:
        logging.error(f"Error occurred during {DATA_INGESTION_STAGE_NAME}: {e}")
        raise e


PREPROCESSING_STAGE_NAME = "Data Preprocessing Stage"


def run_data_preprocessing():
    """
    Runs the data preprocessing stage of the pipeline.
    """
    try:
        logging.info(f"Starting {PREPROCESSING_STAGE_NAME}.")
        preprocessing_pipeline = PreprocessingPipeline()
        preprocessing_pipeline.main()
        logging.info(f"{PREPROCESSING_STAGE_NAME} completed successfully.")
    except Exception as e:
        logging.error(f"Error occurred during {PREPROCESSING_STAGE_NAME}: {e}")
        raise e


def main():
    """
    Main function to run all stages of the pipeline.
    """
    try:
        run_data_ingestion()
        run_data_preprocessing()

    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}")
        raise e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
