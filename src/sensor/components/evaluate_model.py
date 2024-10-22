import os
import logging
import pandas as pd
import mlflow
from sensor.config.configuration import Configuration
from sensor.entity.config_entity import ModelConfig
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class EvaluateModel:
    def __init__(self, model_config: ModelConfig, data: pd.DataFrame, model_path: str):
        """
        Initializes the EvaluateModel class.

        :param model_config: Model configuration.
        :param data: Preprocessed data for evaluation.
        :param model_path: Path to the saved trained model.
        """
        self.model_config = model_config
        self.data = data
        self.model_path = model_path  # Path to the saved model
        self.model = None

    def load_model(self):
        """
        Loads the trained model from the specified path.
        """
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")

            self.model = load_model(self.model_path)
            logging.info("Model loaded successfully for evaluation.")

        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def evaluate(self):
        """
        Evaluates the model using the preprocessed data and logs metrics to MLflow.
        """
        try:
            # Load the model
            self.load_model()

            # Separate features and target from the test data
            X_test = self.data.drop('target', axis=1)
            y_test = self.data['target']
            y_encoded_test = pd.get_dummies(y_test)

            # Make predictions
            y_pred = self.model.predict(X_test)

            # Get the predicted classes (assuming one-hot encoded target)
            y_pred_classes = y_pred.argmax(axis=1)
            y_test_classes = y_encoded_test.values.argmax(axis=1)

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test_classes, y_pred_classes)
            conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
            report = classification_report(y_test_classes, y_pred_classes)

            # Log metrics to MLflow
            mlflow.start_run(run_name="GasSensor_Model_Evaluation")

            mlflow.log_param("model_name", os.path.basename(self.model_path))
            mlflow.log_metric("accuracy", accuracy)

            # Log confusion matrix and classification report to MLflow as artifacts
            mlflow.log_text(str(conf_matrix), "confusion_matrix.txt")
            mlflow.log_text(report, "classification_report.txt")

            # Log the model to MLflow
            mlflow.keras.log_model(self.model, "model")

            mlflow.end_run()

            logging.info(f"Model evaluation completed. Accuracy: {accuracy}")

        except Exception as e:
            mlflow.end_run(status='FAILED')
            logging.error(f"Error during model evaluation: {e}")
            raise


def evaluate_model():
    """
    Evaluate the model using the configurations and data.
    """
    try:
        # Load configuration
        config = Configuration()

        # Get model configuration as a dictionary
        model_config_dict = config.get_model_config()

        # Create an instance of ModelConfig from the dictionary
        model_config = ModelConfig.from_dict(model_config_dict)

        # Load preprocessed data for evaluation
        preprocessed_data_path = os.path.join("artifacts/preprocessed", "preprocessed_data.csv")
        data = pd.read_csv(preprocessed_data_path)

        # Define the path to the saved model (adjust if necessary)
        model_path = os.path.join("artifacts/training", "gas_classification_model_final.keras")

        # Instantiate the EvaluateModel class
        evaluator = EvaluateModel(model_config=model_config, data=data, model_path=model_path)

        # Evaluate the model
        evaluator.evaluate()

    except Exception as e:
        logging.error(f"An error occurred in the evaluate_model function: {e}")
