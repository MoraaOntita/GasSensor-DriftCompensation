import os
import logging
import tensorflow as tf
from tensorflow.keras import layers, models
from sensor.utils.common import create_required_directories
from sensor.config.configuration import Configuration
from sensor.entity.config_entity import ModelConfig


class PrepareBaseModel:
    def __init__(self, config: ModelConfig):
        """
        Initialize PrepareBaseModel with configuration.

        :param config: ModelConfig instance containing model configuration.
        """
        self.config = config
        self.model = None
        logging.info("PrepareBaseModel initialized with configuration.")

    def get_base_model(self):
        """
        Build and return the base model for gas classification.
        """
        try:
            # Ensure input_shape is a tuple
            input_shape = tuple(self.config.input_shape)  
            self.model = self.build_gas_classification_model(input_shape)
            self.save_model(path=self.config.save_dir, model=self.model)
            logging.info("Base model built and saved successfully.")
            
            return self.model
        
        except Exception as e:
            logging.error(f"Error in get_base_model: {e}")
            raise


    def build_gas_classification_model(self, input_shape):
        """
        Build the gas classification neural network architecture.

        :param input_shape: Shape of the input data.
        :return: Compiled Keras model.
        """
        try:
            # Input Layer
            inputs = layers.Input(shape=input_shape)

            # Hidden Layers
            x = layers.Dense(128, activation='relu')(inputs)
            x = layers.Dropout(0.3)(x)  # Regularization
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(32, activation='relu')(x)

            # Output Layer for Gas Classification
            classification_output = layers.Dense(6, activation='softmax', name='gas_classification')(x)

            # Create the model
            model = models.Model(inputs=inputs, outputs=classification_output)

            # Compile the model
            model.compile(optimizer=self.config.optimizer, 
                          loss=self.config.classification_loss,
                          metrics=[self.config.classification_metric])

            logging.info("Gas classification model built successfully.")
            return model

        except Exception as e:
            logging.error(f"Error in build_gas_classification_model: {e}")
            raise

    def save_model(self, path: str, model: tf.keras.Model):
        """
        Save the model to the specified path.

        :param path: Path to save the model (should include filename and extension).
        :param model: Keras model to save.
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Add a filename and extension to the path if it's just a directory
            if os.path.isdir(path):
                path = os.path.join(path, "gas_classification_model.keras")

            model.save(path)
            logging.info(f"Model saved successfully at {path}")
        except Exception as e:
            logging.error(f"Error saving the model: {e}")
            raise


def prepare_base_model():
    """
    Prepare and save the base model.
    """
    try:
        # Load configuration
        config = Configuration()
        model_config_dict = config.get_model_config()  # Get the model config as a dictionary

        # Convert dictionary to ModelConfig instance
        model_config = ModelConfig.from_dict(model_config_dict)

        # Create necessary directories for saving the model
        create_required_directories()

        # Instantiate PrepareBaseModel class with ModelConfig instance
        prepare_model = PrepareBaseModel(config=model_config)

        # Build and save the base model
        prepare_model.get_base_model()

    except Exception as e:
        logging.error(f"Error in prepare_base_model: {e}")

