import sys
import importlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from .base_evaluator import BaseModelEvaluator

class ClassificationModelEvaluator(BaseModelEvaluator):
    """Evaluator for classification tasks."""

    def evaluate_model(self, project):
        try:
            sys.path.append(project['directories']['root'])

            # Dynamically import scripts
            model_module = importlib.import_module('model_training')
            data_processor = importlib.import_module('data_preprocessing')

            # Preprocess data and split
            processed_data = data_processor.preprocess(project['dataset'])
            X_train, X_test, y_train, y_test = train_test_split(
                processed_data['features'],
                processed_data['labels'],
                test_size=0.2,
                random_state=42
            )

            # Train and evaluate
            model = model_module.train_model(X_train, y_train)
            predictions = model.predict(X_test)

            return {
                'accuracy': accuracy_score(y_test, predictions),
                'precision': precision_score(y_test, predictions, average='weighted'),
                'recall': recall_score(y_test, predictions, average='weighted'),
                'f1_score': f1_score(y_test, predictions, average='weighted'),
            }
        except Exception as e:
            logging.error(f"Error in classification evaluation: {e}")
            return {"error": str(e)}
