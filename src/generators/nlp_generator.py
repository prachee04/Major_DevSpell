import os
import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


class NLPGenerator:
    def __init__(self, groq_api_key):
        """Initialize with Groq API"""
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-70b-versatile",
            temperature=0.7,
        )

    def _create_chain(self, prompt_template):
        """Create a simple LLMChain"""
        return LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(prompt_template),
        )

    def _preprocess_dataset(self, dataset):
        """Preprocess the dataset: Either a pandas DataFrame or uploaded file"""
        # If dataset is a pandas DataFrame, return it as-is
        if isinstance(dataset, pd.DataFrame):
            return dataset

        # If dataset is a string (file path), process it
        if isinstance(dataset, str):
            if not os.path.exists(dataset):
                raise FileNotFoundError(f"Dataset file not found: {dataset}")

            file_ext = os.path.splitext(dataset)[1].lower()
            try:
                if file_ext == ".csv":
                    return pd.read_csv(dataset)
                elif file_ext == ".json":
                    return pd.read_json(dataset)
                elif file_ext in [".xls", ".xlsx"]:
                    return pd.read_excel(dataset)
                else:
                    raise ValueError(f"Unsupported file type: {file_ext}")
            except Exception as e:
                raise ValueError(f"Error reading dataset: {str(e)}")

        # Handle file-like objects (e.g., Streamlit uploads)
        if hasattr(dataset, "name"):
            file_ext = os.path.splitext(dataset.name)[1].lower()
            try:
                if file_ext == ".csv":
                    return pd.read_csv(dataset)
                elif file_ext == ".json":
                    return pd.read_json(dataset)
                elif file_ext in [".xls", ".xlsx"]:
                    return pd.read_excel(dataset)
                else:
                    raise ValueError(f"Unsupported file type: {file_ext}")
            except Exception as e:
                raise ValueError(f"Error reading file-like object: {str(e)}")

        raise TypeError("Dataset must be a file path (CSV/JSON), DataFrame, or file-like object")

    def _generate_code(self, prompt_template, **kwargs):
        """Generate code using LLMChain with error handling"""
        try:
            # Create the prompt template with explicit input variables
            prompt = PromptTemplate.from_template(prompt_template)
            
            # Ensure all input variables have values
            input_vars = prompt.input_variables
            inputs = {}
            
            for var in input_vars:
                # Use kwargs if available, otherwise use a default placeholder
                inputs[var] = kwargs.get(var, f"Default {var} content")
            
            # Create and run the chain
            chain = LLMChain(llm=self.llm, prompt=prompt)
            return chain.run(**inputs)
        
        except Exception as e:
            print(f"Error generating code: {e}")
            # Fallback to a generic template if specific generation fails
            return f"# Error generating code\n# {str(e)}\n\n# Placeholder code"

    def _determine_nlp_task(self, df):
        """Determine NLP task based on dataset"""
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        if len(text_columns) == 0:
            raise ValueError("No text columns found for NLP task")
        
        # Simple heuristics to determine task
        if any('label' in col.lower() for col in df.columns):
            return "classification"
        elif any('sentiment' in col.lower() for col in text_columns):
            return "sentiment_analysis"
        elif len(text_columns) > 1:
            return "text_matching"
        else:
            return "text_generation"

    def _generate_project_structure(self, project_name):
        """Generate project directory structure"""
        base_dir = os.path.join(os.getcwd(), project_name)
        dirs = {
            "root": base_dir,
            "src": os.path.join(base_dir, "src"),
            "data": os.path.join(base_dir, "data"),
            "docs": os.path.join(base_dir, "docs"),
            "results": os.path.join(base_dir, "results")
        }
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        return dirs

    def generate(self, dataset):
        """Generate complete NLP project"""
        df = self._preprocess_dataset(dataset)
        project_name = f"nlp_project_{np.random.randint(1000, 9999)}"
        project_dirs = self._generate_project_structure(project_name)

        # Ensure the results folder exists
        results_dir = project_dirs["results"]
        os.makedirs(results_dir, exist_ok=True)

        nlp_task = self._determine_nlp_task(df)

        # Generate all code components
        code_files = {}

        # Data Preprocessing Script
        preprocessing_prompt = """
        Create a Python script for NLP data preprocessing.
        Dataset columns: {columns}
        NLP Task: {task}
        Include:
        - Text cleaning
        - Tokenization
        - Stop word removal
        - Vectorization techniques
        """
        code_files["data_preprocessing.py"] = self._generate_code(
            preprocessing_prompt, 
            columns=str(list(df.columns)), 
            task=nlp_task
        )

        # Model Training Script
        model_prompt = """
        Create a machine learning model for {task} task.
        Dataset columns: {columns}
        Include:
        - Model architectures
        - Training pipeline
        - Hyperparameter tuning
        """
        code_files["model_training.py"] = self._generate_code(
            model_prompt,
            task=nlp_task,
            columns=str(list(df.columns))
        )

        # Evaluation Script
        eval_prompt = """
        Create an evaluation script for NLP models.
        Include metrics:
        - Accuracy
        - Precision
        - Recall
        - F1-score
        - Confusion Matrix
        """
        code_files["model_evaluation.py"] = self._generate_code(eval_prompt)

        # Inference Script
        inference_prompt = """
        Create an inference script for deployed NLP model.
        NLP Task: {task}
        Include:
        - Model loading
        - Preprocessing
        - Prediction
        """
        code_files["model_inference.py"] = self._generate_code(
            inference_prompt, 
            task=nlp_task
        )

        # Write files to the results folder
        for filename, content in code_files.items():
            file_path = os.path.join(results_dir, filename)
            with open(file_path, "w") as f:
                f.write(content)

        # Generate project report
        report_content = f"""
# NLP Project

## Project Overview
Generated Files:
{', '.join(code_files.keys())}

## NLP Task Details
- Task: {nlp_task}
- Comprehensive preprocessing and modeling

## Next Steps
1. Review generated scripts
2. Train and validate models
3. Fine-tune performance
"""
        with open(os.path.join(project_dirs["docs"], "project_report.md"), "w") as f:
            f.write(report_content)

        return {
            "project_name": project_name,
            # "nlp_type": nlp_type,
            "directories": project_dirs,
            "code_files": code_files,
        }