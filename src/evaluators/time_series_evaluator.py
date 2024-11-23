import sys
import importlib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from .base_evaluator import BaseModelEvaluator

class TimeSeriesModelEvaluator(BaseModelEvaluator):
    """Evaluator for time series analysis tasks."""

    def evaluate_model(self, project):
        try:
            sys.path.append(project['directories']['root'])

            # Dynamically import scripts
            model_module = importlib.import_module('model_training')
            data_processor = importlib.import_module('data_preprocessing')

            # Preprocess data
            processed_data = data_processor.preprocess(project['dataset'])

            # Train and evaluate
            model = model_module.train_model(processed_data['features'], processed_data['labels'])
            predictions = model.predict(processed_data['features'])

            return {
                'mse': mean_squared_error(processed_data['labels'], predictions),
                'mae': mean_absolute_error(processed_data['labels'], predictions),
            }
        except Exception as e:
            logging.error(f"Error in time series evaluation: {e}")
            return {"error": str(e)}
