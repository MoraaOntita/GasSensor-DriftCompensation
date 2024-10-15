import logging
from sensor.config.configuration import Configuration
from sensor.components.prepare_base_model import PrepareBaseModel
from sensor.entity.config_entity import ModelConfig

# Define the stage name for logging
STAGE_NAME = "Base Model Preparation Stage"

class PrepareBaseModelPipeline:
    def __init__(self):
        """
        Initializes the pipeline with configuration and model preparation components.
        """
        self.config = Configuration()
        # Get model configuration as a dictionary
        model_config_dict = self.config.get_prepare_base_model_config()  # Ensure this returns a dict
        # Convert the dictionary to a ModelConfig instance
        self.model_config = ModelConfig.from_dict(model_config_dict)  # Use from_dict method

    def main(self):
        """
        Main function that runs the entire model preparation pipeline.
        """
        prepare_model = PrepareBaseModel(config=self.model_config)

        # Prepare the base model
        prepare_model.get_base_model()

if __name__ == '__main__':
    try:
        # Logging the start of the stage
        logging.info(f"*******************")
        logging.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        
        # Create and run the pipeline
        prepare_base_model_pipeline = PrepareBaseModelPipeline()
        prepare_base_model_pipeline.main()

        # Logging the successful completion of the stage
        logging.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    
    except Exception as e:
        logging.exception(e)
        raise e
