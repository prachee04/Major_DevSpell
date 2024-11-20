import os
import io
import pandas as pd
from abc import ABC, abstractmethod

class BaseGenerator(ABC):
    def __init__(self, output_dir='results/generated_projects', groq_api_key=None):
        """
        Initialize base generator with output directory and Groq API key
        
        Args:
            output_dir (str): Directory to save generated projects
            groq_api_key (str): API key for Groq API, optional
        """
        self.output_dir = output_dir
        self.groq_api_key = groq_api_key
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

    def _preprocess_dataset(self, dataset):
        """
        Preprocess input dataset supporting multiple file types.
        
        Args:
            dataset (str or pd.DataFrame or file-like object): Input dataset
        
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        # If dataset is a pandas DataFrame, return it as-is
        if isinstance(dataset, pd.DataFrame):
            return dataset
        
        # If dataset is a string (file path), process it
        if isinstance(dataset, str):
            if not os.path.exists(dataset):
                raise FileNotFoundError(f"Dataset file not found: {dataset}")
            file_ext = os.path.splitext(dataset)[1].lower()
            try:
                if file_ext == '.csv':
                    return pd.read_csv(dataset)
                elif file_ext == '.json':
                    return pd.read_json(dataset)
                else:
                    raise ValueError(f"Unsupported file type: {file_ext}")
            except Exception as e:
                raise ValueError(f"Error reading dataset: {str(e)}")
        
        # Handle Streamlit uploaded file (file-like object)
        if hasattr(dataset, 'name'):
            file_ext = os.path.splitext(dataset.name)[1].lower()
            try:
                # For CSV files
                if file_ext == '.csv':
                    return pd.read_csv(io.BytesIO(dataset.getvalue()))
                
                # For JSON files
                elif file_ext == '.json':
                    return pd.read_json(io.BytesIO(dataset.getvalue()))
                
                # For Excel files
                elif file_ext in ['.xls', '.xlsx']:
                    return pd.read_excel(io.BytesIO(dataset.getvalue()))
                
                else:
                    raise ValueError(f"Unsupported file type: {file_ext}")
            except Exception as e:
                raise ValueError(f"Error reading Streamlit uploaded file: {str(e)}")
        
        raise TypeError("Dataset must be a file path, pandas DataFrame, or Streamlit uploaded file")
    
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
        
        for subdir in subdirs.values():
            os.makedirs(subdir, exist_ok=True)
        
        with open(os.path.join(project_path, 'README.md'), 'w') as f:
            f.write(f"# {project_name}\n\n## Project Overview\n")
        
        return {
            'root': project_path,
            **subdirs
        }
