import logging
from src.sensor.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.sensor.pipeline.stage_02_preprocessing import PreprocessingPipeline
from src.sensor.pipeline.stage_03_prepare_base_model import PrepareBaseModelPipeline
from src.sensor.pipeline.stage_04_train_model import TrainModelPipeline

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

BASE_MODEL_PREPARATION_STAGE_NAME = "Base Model Preparation Stage"

def run_base_model_preparation():
    """
    Runs the base model preparation stage of the pipeline.
    """
    try:
        logging.info(f"Starting {BASE_MODEL_PREPARATION_STAGE_NAME}.")
        prepare_base_model_pipeline = PrepareBaseModelPipeline()  # Create an instance of the pipeline
        prepare_base_model_pipeline.main()  # Call the main method to run it
        logging.info(f"{BASE_MODEL_PREPARATION_STAGE_NAME} completed successfully.")
    except Exception as e:
        logging.error(f"Error occurred during {BASE_MODEL_PREPARATION_STAGE_NAME}: {e}")
        raise e

TRAIN_MODEL_STAGE_NAME = "Model Training Stage"

def run_model_training():
    """
    Runs the model training stage of the pipeline.
    """
    try:
        logging.info(f"Starting {TRAIN_MODEL_STAGE_NAME}.")
        train_model_pipeline = TrainModelPipeline()  # Create an instance of the training pipeline
        train_model_pipeline.main()  # Call the main method to run it
        logging.info(f"{TRAIN_MODEL_STAGE_NAME} completed successfully.")
    except Exception as e:
        logging.error(f"Error occurred during {TRAIN_MODEL_STAGE_NAME}: {e}")
        raise e

def main():
    """
    Main function to run all stages of the pipeline.
    """
    try:
        run_data_ingestion()
        run_data_preprocessing()
        run_base_model_preparation()  # Run the base model preparation stage
        run_model_training()  # Run the model training stage

    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}")
        raise e

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
