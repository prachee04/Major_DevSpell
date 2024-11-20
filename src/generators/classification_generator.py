import os
import numpy as np
import pandas as pd
from typing import Dict, Any

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from langchain_core.runnables import Runnable

class ClassificationProjectSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    project_name: str = Field(description="Name of the classification project")
    preprocessing_code: str = Field(description="Python code for data preprocessing")
    feature_engineering_code: str = Field(description="Python code for feature engineering")
    model_training_code: str = Field(description="Python code for model training")
    evaluation_code: str = Field(description="Python code for model evaluation")

class ClassificationGenerator:
    def __init__(self, groq_api_key: str):
        """
        Initialize the Classification Generator with Groq API
        
        :param groq_api_key: API key for Groq
        """
        self.llm = ChatGroq(
            temperature=0.3, 
            model_name="llama3-70b-8192", 
            groq_api_key=groq_api_key
        )
    
    def generate(self, dataset, project_type='classification') -> Dict[str, Any]:
        """
        Generate Classification project
        
        :param dataset: Uploaded dataset file
        :param project_type: Type of classification project
        :return: Generated project details
        """
        # Load and preprocess dataset
        df = self._load_dataset(dataset)
        
        # Determine classification type
        classification_type = self._determine_classification_type(df)
        
        # Create project name
        project_name = f"classification_{classification_type}_{np.random.randint(1000, 9999)}"
        
        # Generate project structure
        project_dirs = self._generate_project_structure(project_name)
        
        # Prepare prompt for code generation
        prompt = self._create_generation_prompt(df, classification_type)
        
        # Generate project code
        try:
            # Get LLM response
            llm_response = self.llm.predict(prompt)
            
            # Manually parse the response (you might need to adjust this)
            project_spec = self._parse_project_spec(llm_response, project_name)
        except Exception as e:
            print(f"Error generating project: {e}")
            raise
        
        # Save generated code files
        code_files = self._save_generated_code(project_dirs, project_spec)
        
        # Generate project report
        self._generate_project_report(project_dirs, code_files)
        
        return {
            'project_name': project_name,
            'project_type': 'Classification',
            'classification_type': classification_type,
            'directories': project_dirs,
            'code_files': code_files
        }
    
    def _create_generation_prompt(self, df, classification_type):
        """
        Create a detailed prompt for code generation
        
        :param df: Input DataFrame
        :param classification_type: Type of classification
        :return: Prompt string
        """
        return f"""Generate a complete machine learning classification project with the following specifications:

Project Requirements:
- Classification Type: {classification_type}
- Dataset Features: {', '.join(df.columns.drop('target').tolist())}
- Target Variable: target
- Number of Classes: {len(df['target'].unique())}

Please generate the following Python scripts:
1. data_preprocessing.py: Handle missing values, encode categorical variables, scale features
2. feature_engineering.py: Perform feature selection and engineering
3. model_training.py: Train appropriate classification models
4. evaluation.py: Implement model evaluation metrics

Provide complete, production-ready code that follows best practices. Include necessary imports, comments, and error handling.

Output Format:
- Each script should be a complete, runnable Python script
- Use scikit-learn for preprocessing and modeling
- Include type hints and docstrings
- Handle potential errors gracefully
"""
    
    def _parse_project_spec(self, llm_response: str, project_name: str) -> ClassificationProjectSpec:
        """
        Manually parse the LLM response into a project specification
        
        :param llm_response: Raw LLM response
        :param project_name: Name of the project
        :return: Parsed project specification
        """
        # This is a very basic parsing - you'll likely need to improve this
        return ClassificationProjectSpec(
            project_name=project_name,
            preprocessing_code=self._extract_script(llm_response, 'data_preprocessing.py'),
            feature_engineering_code=self._extract_script(llm_response, 'feature_engineering.py'),
            model_training_code=self._extract_script(llm_response, 'model_training.py'),
            evaluation_code=self._extract_script(llm_response, 'evaluation.py')
        )
    
    def _extract_script(self, response: str, filename: str) -> str:
        """
        Extract a specific script from the LLM response
        
        :param response: Full LLM response
        :param filename: Name of the script to extract
        :return: Extracted script content
        """
        # Basic script extraction - you'll need more robust parsing
        import re
        
        script_pattern = rf"```python\n.*?{filename}.*?\n(.*?)```"
        match = re.search(script_pattern, response, re.DOTALL)
        
        return match.group(1) if match else f"# Could not extract {filename}"
    
    def _load_dataset(self, dataset):
        """
        Load dataset from CSV or JSON
        
        :param dataset: Uploaded dataset file
        :return: Processed DataFrame
        """
        if dataset is None:
            raise ValueError("No dataset provided")
        
        # Determine file type
        file_extension = os.path.splitext(dataset.name)[1].lower()
        
        # Read dataset
        if file_extension == '.csv':
            df = pd.read_csv(dataset)
        elif file_extension in ['.json', '.jsonl']:
            df = pd.read_json(dataset)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Validate dataset
        if 'target' not in df.columns:
            raise ValueError("Dataset must contain a 'target' column")
        
        return df
    
    # ... (rest of the methods remain the same as in previous implementations)
    # Include _save_generated_code, _determine_classification_type, 
    # _generate_project_structure, and _generate_project_report methods 
    # from the previous implementation