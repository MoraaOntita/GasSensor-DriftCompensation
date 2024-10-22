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
    input_shape: tuple
    save_dir: str
    optimizer: str
    classification_loss: str
    drift_loss: str
    classification_metric: str
    drift_metric: str

    @classmethod
    def from_dict(cls, config_dict):
        """
        Create ModelConfig instance from a dictionary.
        Fallback logic included for missing keys.
        """
        return cls(
            input_shape=tuple(config_dict.get('input_shape', [128])),  # Fallback to (128,)
            save_dir=config_dict.get('save_dir', 'artifacts/prepared_model/'),
            optimizer=config_dict.get('optimizer', 'adam'),
            classification_loss=config_dict.get('classification_loss', 'categorical_crossentropy'),
            drift_loss=config_dict.get('drift_loss', 'mean_squared_error'),
            classification_metric=config_dict.get('classification_metric', 'accuracy'),
            drift_metric=config_dict.get('drift_metric', 'mse'),
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
        
@dataclass
class MLflowConfig:
    tracking_uri: str
    username: str
    password: str

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create MLflowConfig from a dictionary."""
        return cls(
            tracking_uri=config_dict['tracking_uri'],
            username=config_dict['username'],
            password=config_dict['password']
        )