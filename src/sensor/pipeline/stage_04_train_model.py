import logging
import pandas as pd
from sensor.config.configuration import Configuration
from sensor.components.train_model import TrainModel
from sensor.entity.config_entity import ModelConfig

# Define the stage name for logging
STAGE_NAME = "Model Training Stage"

class TrainModelPipeline:
    def __init__(self):
        """
        Initializes the pipeline with configuration and model training components.
        """
        self.config = Configuration()
        model_config_dict = self.config.get_model_config()  # This should return a dict
        
        # Log the configuration for debugging
        logging.info(f"Model config dictionary: {model_config_dict}, Type: {type(model_config_dict)}")

        # Ensure that model_config_dict is indeed a dictionary
        if not isinstance(model_config_dict, dict):
            logging.error(f"Model config is not a dictionary: {model_config_dict}, Type: {type(model_config_dict)}")
            raise ValueError("Expected model_config_dict to be a dictionary.")

        # Convert the dictionary to a ModelConfig instance
        self.model_config = ModelConfig.from_dict(model_config_dict)  # Use from_dict method
        
        # Get training parameters
        self.training_config = self.config.get_training_params()  # Retrieve training parameters

    def main(self):
        """
        Main method to execute the model training process.
        """
        try:
            # Load your training data
            data_path = self.config.get_training_data_path()  # Make sure this method exists
            data = pd.read_csv(data_path)  # Load the data
            
            # Log before starting training
            logging.info(f"Starting model training with config: {self.model_config} and training config: {self.training_config}")

            # Create an instance of TrainModel with the required arguments
            train_model = TrainModel(self.model_config, self.training_config, data)  # Provide the required parameters
            
            # Start the model training
            train_model.train()  # Ensure that the TrainModel class has a train() method
            
            logging.info(f"Model training completed successfully.")
        except Exception as e:
            logging.error(f"Error occurred during model training: {e}")
            raise e


if __name__ == '__main__':
    try:
        # Logging the start of the stage
        logging.info(f"*******************")
        logging.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        
        # Create and run the pipeline
        train_model_pipeline = TrainModelPipeline()
        train_model_pipeline.main()

        # Logging the successful completion of the stage
        logging.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    
    except Exception as e:
        logging.exception(e)
        raise e
