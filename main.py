import logging
from src.sensor.pipeline.stage_01_data_ingestion import start_data_ingestion

STAGE_NAME = "Data Ingestion stage"

def main():
    try:
        logging.info("Starting the data ingestion process.")
        start_data_ingestion()
        logging.info("Data ingestion process completed.")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise e

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
