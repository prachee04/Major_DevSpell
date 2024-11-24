import os
import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import chardet
import io
import re



def detect_encoding(file):
    with open(file, 'rb') as f:
        raw_data = f.read(10000)  # Read a chunk of the file
    result = chardet.detect(raw_data)
    return result['encoding']



class NLPGenerator:
    def __init__(self, groq_api_key, model, name):
        """Initialize with Groq API"""
        self.name= name
        self.model=model
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model,
            temperature=0.7,
        )

    def _create_chain(self, prompt_template):
        """Create a simple LLMChain"""
        return LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(prompt_template),
        )

    def _preprocess_dataset(self, dataset):
        """
        Preprocess the dataset: Either a pandas DataFrame or uploaded file.
        Handles file encoding issues.
        """
        # If dataset is a pandas DataFrame, return it as-is
        if isinstance(dataset, pd.DataFrame):
            return dataset

        # If dataset is a string (file path), process it
        if isinstance(dataset, str):
            if not os.path.exists(dataset):
                raise FileNotFoundError(f"Dataset file not found: {dataset}")

            # Detect encoding if the file cannot be read with UTF-8
            encoding = detect_encoding(dataset)

            file_ext = os.path.splitext(dataset)[1].lower()
            try:
                if file_ext == ".csv":
                    return pd.read_csv(dataset, encoding=encoding)
                elif file_ext == ".json":
                    return pd.read_json(dataset, encoding=encoding)
                elif file_ext in [".xls", ".xlsx"]:
                    return pd.read_excel(dataset)
                else:
                    raise ValueError(f"Unsupported file type: {file_ext}")
            except Exception as e:
                raise ValueError(f"Error reading dataset: {str(e)}")

        # Handle file-like objects (e.g., Streamlit uploads)
        if hasattr(dataset, "name"):
            file_ext = os.path.splitext(dataset.name)[1].lower()
            file_bytes = dataset.getvalue()
            encoding = chardet.detect(file_bytes)['encoding']

            try:
                if file_ext == ".csv":
                    return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
                elif file_ext == ".json":
                    return pd.read_json(io.BytesIO(file_bytes), encoding=encoding)
                elif file_ext in [".xls", ".xlsx"]:
                    return pd.read_excel(io.BytesIO(file_bytes))
                else:
                    raise ValueError(f"Unsupported file type: {file_ext}")
            except Exception as e:
                raise ValueError(f"Error reading file-like object: {str(e)}")

        raise TypeError("Dataset must be a file path (CSV/JSON), DataFrame, or file-like object")
    import re

    def _sanitize_output(self, text):
        """
        Extract and return only valid Python code from the text.
        Removes any non-Python content, including markdown, extra formatting,
        and irrelevant text.
        """
        # Regular expression to match Python code blocks
        python_code_blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
        
        if python_code_blocks:
            # Concatenate all detected Python code blocks
            sanitized_code = "\n".join(python_code_blocks)
        else:
            # If no explicit Python code blocks, assume the entire text might be code
            # but filter out anything obviously not Python (like markdown headers)
            sanitized_code = re.sub(r"[^a-zA-Z0-9_#:\n\(\)\[\]\{\}.,=+\-*\/<>%&|! ]", "", text)
        
        return sanitized_code.strip()


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
            return self._sanitize_output(chain.run(**inputs))


        
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
        
        base_dir = os.path.join(os.getcwd(), "results", self.name, self.model)
        dirs = {
            "root": base_dir,
            "src": os.path.join(base_dir, "src"),
            "dataset": os.path.join(base_dir, "dataset"),  # Added dataset directory
            "docs": os.path.join(base_dir, "docs"),
            "results": os.path.join(base_dir, "results")
        }
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        return dirs


    def generate(self, dataset):
        """Generate complete NLP project"""
        df = self._preprocess_dataset(dataset)

        project_dirs = self._generate_project_structure(self.name)

        # Save dataset to CSV in the dataset directory
        dataset_path = os.path.join(project_dirs["results"], "test.csv")
        df.to_csv(dataset_path, index=False)

        # Ensure the results folder exists
        results_dir = project_dirs["results"]
        os.makedirs(results_dir, exist_ok=True)

        nlp_task = self._determine_nlp_task(df)

        # Generate all code components
        code_files = {}

        # Data Preprocessing Script
        preprocessing_prompt = """
        You are an expert NLP data scientist. Write a Python script for data preprocessing. The dataset is named 'test.csv' and contains the columns: {columns}. The task is: {task}. Include:
1. Text cleaning (e.g., lowercasing, removing special characters).
2. Tokenization.
3. Stop word removal.
4. Lemmatization.
5. Vectorization using TF-IDF and CountVectorizer.

Encapsulate the preprocessing steps in functions. Include a `run` function that accepts the dataset path as input.
        """
        code_files["data_preprocessing.py"] = self._generate_code(
            preprocessing_prompt, 
            columns=str(list(df.columns)), 
            task=nlp_task
        )

        model_prompt = """
You are a seasoned machine learning engineer with expertise in designing and implementing end-to-end ML pipelines. Your task is to generate a Python script for building a machine learning model.

Specifications:

    1. The task is: {task}.
    2. The dataset contains columns: {columns}.
    3. The dataset should be loaded from "/test.csv".
    4. The script must include:
        - Implementation of one or more model architectures suited to the task.
        - A training pipeline with data preprocessing, splitting, model training, and evaluation.
        - Hyperparameter tuning using grid search, random search, or a modern library like Optuna or Hyperopt.
        - Save the trained model to './results/trained_model.pkl'.
    5. The script should be runnable directly from the command line.
    6. Only write the python script. Anything else should be included as python comments.

Code Structure Requirements:
    1. Include a main() function that orchestrates the entire training process.
    2. Use proper error handling for file operations.
    

Constraints:

    1. Provide only the Python script; any explanation or non-code instructions should appear as comments within the code.
    2. Do not include any start or end markers for the code (e.g., ```python or ```).
    3. The code must be modular, clean, and include function definitions for key steps to ensure reusability.
    4. All file paths should be relative to the script's location in the results directory.

Additional Notes:

    1. Select appropriate machine learning libraries (e.g., scikit-learn, TensorFlow, PyTorch) based on the task's requirements.
    2. Optimize for clarity and usability, making the script easy to understand and adapt.
    3. Include clear logging statements to track the training progress.
    4. Use relative imports for any custom modules from data_preprocessing.py.
    5. Do not include anything other than python code.

"""

        code_files["model_training.py"] = self._generate_code(
            model_prompt,
            task=nlp_task,
            columns=str(list(df.columns))
        )

        # Evaluation Script
        eval_prompt = """
        You are an NLP evaluation specialist. Write a Python script for evaluating NLP models. The script should:

1. Compute Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
2. Accept true labels and predicted labels as inputs, with an optional parameter for class labels.
3. Encapsulate each metric calculation in reusable functions.
4. Use scikit-learn for computations.

Include a `main` block to demonstrate usage with example inputs.

        """
        code_files["model_evaluation.py"] = self._generate_code(eval_prompt)

        # Inference Script
        inference_prompt = """
        You are a deployment and inference engineer. Write a Python script for inference with an NLP model for the task: {task}. The script should include:
1. Loading the trained model.
2. Text preprocessing for inference.
3. Generating predictions.
Ensure modularity with separate functions for preprocessing and inference. The script must be callable independently.

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

    ## Dataset Location
    The dataset is stored in: dataset/test.csv

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
            "project_name": self.name,
            "directories": project_dirs,
            "code_files": code_files,
            "dataset_path": dataset_path
        }