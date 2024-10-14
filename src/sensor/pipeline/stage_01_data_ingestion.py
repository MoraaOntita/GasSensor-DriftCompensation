from sensor.config.configuration import Configuration
from sensor.components.data_ingestion import DataIngestion
from sensor.entity.config_entity import DataIngestionConfig

def start_data_ingestion():
    config = Configuration()
    data_ingestion_config = DataIngestionConfig(**config.get_data_ingestion_config())
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion.initiate_data_ingestion()
