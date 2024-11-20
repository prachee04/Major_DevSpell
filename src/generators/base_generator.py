import os
import pandas as pd
from abc import ABC, abstractmethod

class BaseGenerator(ABC):
    def __init__(self, output_dir='results/generated_projects'):
        """
        Initialize base generator with output directory
        
        Args:
            output_dir (str): Directory to save generated projects
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    @abstractmethod
    def generate(self, dataset, llm):
        """
        Generate ML project for a specific domain
        
        Args:
            dataset (pd.DataFrame): Input dataset
            llm (object): Language model for generation
        
        Returns:
            dict: Generated project details
        """
        pass
    
    import pandas as pd
    import os

    def _preprocess_dataset(self, dataset):
        """
        Preprocess input dataset supporting multiple file types
        
        Args:
            dataset (str or pd.DataFrame): Input dataset
        
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        # If dataset is already a DataFrame, return it
        if isinstance(dataset, pd.DataFrame):
            return dataset
        
        # If dataset is a string (file path)
        if isinstance(dataset, str):
            # Check file exists
            if not os.path.exists(dataset):
                raise FileNotFoundError(f"Dataset file not found: {dataset}")
            
            # Get file extension
            file_ext = os.path.splitext(dataset)[1].lower()
            
            # Read based on file type
            try:
                if file_ext == '.csv':
                    return pd.read_csv(dataset)
                elif file_ext == '.json':
                    return pd.read_json(dataset)
                else:
                    raise ValueError(f"Unsupported file type: {file_ext}")
            except Exception as e:
                raise ValueError(f"Error reading dataset: {str(e)}")
        
        # If not DataFrame or file path
        raise TypeError("Dataset must be a file path or pandas DataFrame")
    
    def _generate_project_structure(self, project_name):
        """
        Create project directory and basic files
        
        Args:
            project_name (str): Name of the project
        
        Returns:
            dict: Project directory paths
        """
        project_path = os.path.join(self.output_dir, project_name)
        os.makedirs(project_path, exist_ok=True)
        
        # Create standard project structure
        subdirs = {
            'data': os.path.join(project_path, 'data'),
            'notebooks': os.path.join(project_path, 'notebooks'),
            'src': os.path.join(project_path, 'src'),
            'tests': os.path.join(project_path, 'tests'),
            'docs': os.path.join(project_path, 'docs')
        }
        
        # Create subdirectories
        for subdir in subdirs.values():
            os.makedirs(subdir, exist_ok=True)
        
        # Create README
        with open(os.path.join(project_path, 'README.md'), 'w') as f:
            f.write(f"# {project_name}\n\n## Project Overview\n")
        
        return {
            'root': project_path,
            **subdirs
        }