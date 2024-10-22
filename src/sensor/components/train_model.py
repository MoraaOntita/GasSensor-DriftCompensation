import os
import logging
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sensor.config.configuration import Configuration
from sensor.entity.config_entity import ModelConfig, DataPreprocessingConfig
from sensor.components.prepare_base_model import PrepareBaseModel
from sensor.utils.common import create_required_directories


class TrainModel:
    def __init__(self, model_config: ModelConfig, training_config: dict, data: pd.DataFrame):
        """
        Initializes the TrainModel class.

        :param model_config: Model configuration.
        :param training_config: Dictionary containing training parameters.
        :param data: Preprocessed data for training.
        """
        self.model_config = model_config
        self.training_config = training_config
        self.data = data
        self.model = None

    def load_data(self):
        """
        Loads training and testing data.

        :return: Tuple of training and testing datasets.
        """
        try:
            # Separate features and target
            X = self.data.drop('target', axis=1)
            y = self.data['target']

            # One-hot encoding for the target variable
            y_encoded = pd.get_dummies(y)

            # Split the dataset into training and testing sets
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def train(self):
        """
        Train the model using the preprocessed data.
        """
        try:
            X_train, X_test, y_train, y_test = self.load_data()

            # Instantiate the model preparation class to load the model
            prepare_model = PrepareBaseModel(config=self.model_config)
            self.model = prepare_model.get_base_model()  # Prepare the model

            # Define callbacks
            early_stopping = EarlyStopping(
                patience=self.training_config['early_stopping_patience'],
                restore_best_weights=self.training_config['restore_best_weights']
            )

            # Train the model
            history = self.model.fit(
                X_train, y_train,
                validation_split=self.training_config['validation_split'],
                epochs=self.training_config['epochs'],
                batch_size=self.training_config['batch_size'],
                callbacks=[early_stopping]
            )

            # Save the trained model to the specified path
            model_save_path = os.path.join("artifacts/training", "gas_classification_model.keras")
            prepare_model.save_model(model_save_path, self.model)

            # Save training history
            self.save_training_history(history)

        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise

    def save_training_history(self, history):
        """
        Save training history to a file for future reference.

        :param history: Training history object returned by model.fit().
        """
        try:
            history_df = pd.DataFrame(history.history)
            history_save_path = os.path.join("artifacts/training", "training_history.csv")
            history_df.to_csv(history_save_path, index=False)
            logging.info(f"Training history saved at: {history_save_path}")
        except Exception as e:
            logging.error(f"Error saving training history: {e}")
            raise


def train_model():
    """
    Train the model using the configurations and data.
    """
    try:
        # Load configuration
        config = Configuration()
        model_config = config.get_model_config()
        training_config = config.get_training_params()

        # Load preprocessed data
        preprocessed_data_path = os.path.join("artifacts/preprocessed", "preprocessed_data.csv")
        data = pd.read_csv(preprocessed_data_path)

        # Create necessary directories for saving the training history and model
        create_required_directories()

        # Instantiate the TrainModel class
        trainer = TrainModel(model_config=model_config, training_config=training_config, data=data)

        # Train the model
        trainer.train()

    except Exception as e:
        logging.error(f"An error occurred in the train_model function: {e}")
