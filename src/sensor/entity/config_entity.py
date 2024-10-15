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
    
    
@dataclass
class ModelConfig:
    input_shape: tuple
    save_dir: str
    optimizer: str
    classification_loss: str
    drift_loss: str
    classification_metric: str
    drift_metric: str

    @classmethod
    def from_dict(cls, config_dict):
        """Create ModelConfig from a dictionary."""
        return cls(
            input_shape=tuple(config_dict['input_shape']),  # Convert to tuple if needed
            save_dir=config_dict['save_dir'],
            optimizer=config_dict['optimizer'],
            classification_loss=config_dict['classification_loss'],
            drift_loss=config_dict['drift_loss'],
            classification_metric=config_dict['classification_metric'],
            drift_metric=config_dict['drift_metric']
        )