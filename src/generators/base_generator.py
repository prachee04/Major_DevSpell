from abc import ABC, abstractmethod

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, dataset, llm):
        """
        Generate ML project for a specific domain
        """
        pass
    
    def _preprocess_dataset(self, dataset):
        """
        Common dataset preprocessing methods
        """
        pass
    
    def _generate_project_structure(self):
        """
        Create project directory and files
        """
        pass