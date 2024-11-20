import os
import pandas as pd
import numpy as np
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from src.generators.base_generator import BaseGenerator
from langchain_groq import ChatGroq

class NLPGenerator(BaseGenerator):
    def __init__(self, groq_api_key=None):
        """
        Initialize the NLPGenerator with an optional API key for Groq.
        
        Args:
            groq_api_key (str, optional): The API key for ChatGroq API. Defaults to None.
        """
        super().__init__()
        
        # Initialize the LLM with ChatGroq API key if provided
        if groq_api_key is None:
            groq_api_key = os.getenv("GROQ_API_KEY")  # Fallback to environment variable if not provided
        self.llm = ChatGroq(api_key=groq_api_key, model="llama-3.1-70b-versatile")

    def _preprocess_dataset(self, dataset):
        """
        Preprocess input dataset for NLP project
        """
        if isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
        elif isinstance(dataset, str):
            if dataset.lower().endswith('.csv'):
                df = pd.read_csv(dataset)
            elif dataset.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(dataset)
            elif dataset.lower().endswith('.json'):
                df = pd.read_json(dataset)
            else:
                raise ValueError(f"Unsupported file type: {dataset}")
        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset)}")
        
        if df.empty:
            raise ValueError("Input dataset is empty")
        
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df

    def _generate_code_with_llm(self, prompt):
        """
        Generate code using the LLM with LangChain
        """
        prompt_template = PromptTemplate(input_variables=["dataset"], template=prompt)
        chain = LLMChain(prompt=prompt_template, llm=self.llm)
        result = chain.run(dataset="Dataset information here.")
        return result

    def _generate_nlp_code(self, df, project_dirs):
        """
        Generate NLP code files using LLM
        """
        preprocessing_prompt = """
        Write a Python script for preprocessing text data for NLP tasks. 
        Ensure that the script handles missing values, tokenization, and basic cleaning. 
        Input: {dataset}
        """
        model_training_prompt = """
        Write a Python script for training an NLP classification model using scikit-learn or PyTorch.
        Include preprocessing steps, model architecture, training loop, and evaluation logic.
        Input: {dataset}
        """
        evaluation_prompt = """
        Write a Python script to evaluate a trained NLP model using metrics like accuracy and F1-score.
        The evaluation should include confusion matrix generation.
        Input: {dataset}
        """
        
        # Use the LLM to generate code
        preprocessing_code = self._generate_code_with_llm(preprocessing_prompt)
        model_training_code = self._generate_code_with_llm(model_training_prompt)
        evaluation_code = self._generate_code_with_llm(evaluation_prompt)
        
        # Save the generated code to files
        code_files = {
            'preprocessing.py': preprocessing_code,
            'model_training.py': model_training_code,
            'evaluation.py': evaluation_code
        }
        
        for filename, code_content in code_files.items():
            file_path = os.path.join(project_dirs['src'], filename)
            with open(file_path, 'w') as f:
                f.write(code_content)
        
        return code_files

    def generate(self, dataset):
        """
        Generate the NLP project with LangChain and LLM
        """
        df = self._preprocess_dataset(dataset)
        
        project_name = f"nlp_project_{np.random.randint(1000, 9999)}"
        project_dirs = self._generate_project_structure(project_name)
        
        # Save the preprocessed dataset
        df.to_csv(os.path.join(project_dirs['data'], 'nlp_data.csv'), index=False)
        
        # Generate NLP project code
        nlp_code = self._generate_nlp_code(df, project_dirs)
        
        return {
            'project_name': project_name,
            'project_type': 'Natural Language Processing',
            'directories': project_dirs,
            'code_files': nlp_code,
            'preprocessed_data': df
        }
