from abc import ABC, abstractmethod

class BaseModelEvaluator(ABC):
    """Abstract base class for model evaluation."""
    
    @abstractmethod
    def evaluate_model(self, project):
        """
        Evaluate a specific model based on the project details.

        Args:
            project (dict): Project details including dataset and generated scripts.

        Returns:
            dict: Evaluation metrics.
        """
        pass
