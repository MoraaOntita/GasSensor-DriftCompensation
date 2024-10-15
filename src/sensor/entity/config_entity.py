from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    root_dir: str
    source_URL: str
    local_data_file: str
    unzip_dir: str


@dataclass
class DataPreprocessingConfig:
    preprocessed_dir: str
    preprocessed_file: str