from dataclasses import dataclass
from typing import Tuple, Dict, Any

@dataclass
class DataIngestionConfig:
    root_dir: str
    source_URL: str
    local_data_file: str
    unzip_dir: str

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create DataIngestionConfig from a dictionary."""
        return cls(
            root_dir=config_dict['root_dir'],
            source_URL=config_dict['source_URL'],
            local_data_file=config_dict['local_data_file'],
            unzip_dir=config_dict['unzip_dir']
        )


@dataclass
class DataPreprocessingConfig:
    preprocessed_dir: str
    preprocessed_file: str
    num_features: int
    feature_range: Tuple[int, int]
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create DataPreprocessingConfig from a dictionary."""
        return cls(
            preprocessed_dir=config_dict['preprocessed_dir'],
            preprocessed_file=config_dict['preprocessed_file'],
            num_features=config_dict['num_features'],
            feature_range=tuple(config_dict['feature_range'])  # Ensure it's a tuple
        )
    
    
@dataclass
class ModelConfig:
    input_shape: Tuple[int, ...]  # Allow for flexible dimensions
    save_dir: str
    optimizer: str
    classification_loss: str
    drift_loss: str
    classification_metric: str
    drift_metric: str

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
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

@dataclass
class ModelTrainingConfig:
    epochs: int
    batch_size: int
    validation_split: float
    early_stopping_patience: int
    restore_best_weights: bool

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create ModelTrainingConfig from a dictionary."""
        return cls(
            epochs=config_dict['epochs'],
            batch_size=config_dict['batch_size'],
            validation_split=config_dict['validation_split'],
            early_stopping_patience=config_dict['early_stopping_patience'],
            restore_best_weights=config_dict['restore_best_weights']
        )