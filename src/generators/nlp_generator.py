from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.runnables import Runnable  
from src.generators.base_generator import BaseGenerator
import os
import pandas as pd
import numpy as np

class GroqRunnable(Runnable):
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model

    def _call(self, inputs: dict):
        # This should match how Groq responds to calls for text generation
        # Make Groq API call here and return the result in the required format.
        # For example:
        result = self.generate_code(inputs["task"])  # Assuming 'task' is the key
        return result

    def generate_code(self, task_description: str) -> str:
        # You would add your Groq API call logic here.
        # For simplicity, let's assume it takes the task_description and calls Groq's API.
        response = groq_api_call(self.api_key, self.model, task_description)
        return response['output']

def groq_api_call(api_key, model, task_description):
    # This is just a placeholder for actual Groq API interaction
    # You need to replace it with actual API request logic
    return {"output": f"Generated code for {task_description} with {model}"}

class NLPGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()
        self.llm = GroqRunnable(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-70b-versatile")

    def _preprocess_dataset(self, dataset):
        """
        Preprocess input dataset for NLP project
        """
        # Dataset preprocessing logic (same as before)
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
        # Code generation logic (same as before)
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
