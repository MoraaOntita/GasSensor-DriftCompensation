import os
import logging
import pandas as pd
from sensor.config.configuration import Configuration
from sensor.entity.config_entity import ModelConfig
from sensor.components.evaluate_model import EvaluateModel 
from sensor.utils.common import create_required_directories 

class EvaluateModelPipeline:
    def __init__(self):
        self.config = Configuration()
        self.model_config = self.config.get_model_config()
        logging.info(f"Model configuration loaded: {self.model_config}")

    def main(self):
        """
        Main function to execute evaluation.
        """
        self.evaluate()

    def evaluate(self):
        """
        The main function to evaluate the model.
        """
        try:
            # Load preprocessed data for evaluation
            preprocessed_data_path = os.path.join("artifacts/preprocessed", "preprocessed_data.csv")
            
            # Ensure the preprocessed data file exists
            if not os.path.exists(preprocessed_data_path):
                logging.error(f"Preprocessed data file does not exist at: {preprocessed_data_path}")
                return

            data = pd.read_csv(preprocessed_data_path)

            # Check if the loaded data is empty
            if data.empty:
                logging.error("Loaded data is empty. Evaluation cannot proceed.")
                return

            # Create necessary directories for saving evaluation results if needed
            create_required_directories()

            # Define the path to the saved model
            model_path = os.path.join("artifacts/training", "gas_classification_model_final.keras")
            
            # Ensure the model file exists
            if not os.path.exists(model_path):
                logging.error(f"Trained model file does not exist at: {model_path}")
                return

            # Instantiate the EvaluateModel class with the model path
            evaluator = EvaluateModel(model_config=self.model_config, data=data, model_path=model_path)

            # Evaluate the model
            evaluator.evaluate()

        except Exception as e:
            logging.error(f"An error occurred in the evaluate_model_pipeline function: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    pipeline = EvaluateModelPipeline()  # Create an instance of the pipeline
    pipeline.main()  # Call the main method to run it
