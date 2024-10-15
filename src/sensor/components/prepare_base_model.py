import os
import logging
import tensorflow as tf
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

    def get_base_model(self):
        """
        Build and save the base model.
        """
        input_shape = self.config.input_shape
        self.model = self.build_multitask_model(input_shape)
        self.save_model(path=self.config.save_dir, model=self.model)

    def build_multitask_model(self, input_shape):
        """
        Build the multi-task neural network architecture.
        :param input_shape: Shape of the input data.
        :return: Compiled Keras model.
        """
        # Input Layer
        inputs = tf.keras.layers.Input(shape=input_shape)

        # Shared Layers
        shared = tf.keras.layers.Dense(64, activation='relu')(inputs)
        shared = tf.keras.layers.Dense(32, activation='relu')(shared)

        # Output for Gas Classification
        classification_output = tf.keras.layers.Dense(6, activation='softmax', name='gas_classification')(shared)

        # Output for Drift Prediction
        drift_output = tf.keras.layers.Dense(1, name='drift_prediction')(shared)

        # Build the model
        model = tf.keras.Model(inputs=inputs, outputs=[classification_output, drift_output])

        # Compile the model
        model.compile(optimizer=self.config.optimizer,
                      loss={
                          'gas_classification': self.config.classification_loss,
                          'drift_prediction': self.config.drift_loss
                      },
                      metrics={
                          'gas_classification': self.config.classification_metric,
                          'drift_prediction': self.config.drift_metric
                      })

        return model

    def save_model(self, path: str, model: tf.keras.Model):
        """
        Save the model to the specified path.
        :param path: Path to save the model (should include filename and extension).
        :param model: Keras model to save.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Add a filename and extension to the path if it's just a directory
        if os.path.isdir(path):
            path = os.path.join(path, "model.keras")  # or "model.h5", depending on your preference

        model.save(path)
        logging.info(f"Model saved successfully at {path}")

    def update_and_save_full_model(self):
        """
        Prepare and save the full model with additional configurations if needed.
        This method can be extended for further customizations.
        """
        self.save_model(path=os.path.join(self.config.save_dir, "updated_model.keras"), model=self.model)

def prepare_base_model():
    """
    Prepare and save the base model.
    """
    # Load configuration
    config = Configuration()
    model_config = config.get_model_config()

    # Create necessary directories for saving the model
    create_required_directories()

    # Instantiate PrepareBaseModel class
    prepare_model = PrepareBaseModel(config=model_config)

    # Build and save the base model
    prepare_model.get_base_model()

    # Optional: Update and save the full model if needed
    prepare_model.update_and_save_full_model()
