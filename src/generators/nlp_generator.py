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
        dataset_path = os.path.join(project_dirs["dataset"], "test.csv")
        df.to_csv(dataset_path, index=False)

        # Ensure the results folder exists
        results_dir = project_dirs["results"]
        os.makedirs(results_dir, exist_ok=True)

        nlp_task = self._determine_nlp_task(df)

        # Generate all code components
        code_files = {}

        # Data Preprocessing Script

        
        preprocessing_prompt = """
You are an experienced Python developer specializing in NLP data preprocessing. Generate a Python script for preprocessing NLP data that follows a clear pipeline architecture.

Specifications:

1. Create a PreprocessingPipeline class with the following methods:
    - _init_(self, config: Dict[str, Any]):
        # Store normalized column names from config
        self.columns = [unidecode(col).strip().lower().replace(' ', '_') for col in config['columns']]
        self.config = config

    - normalize_column_name(self, column: str) -> str:
        # Consistently normalize column names
        return unidecode(str(column)).strip().lower().replace(' ', '_')

    - validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalize all DataFrame column names
        df.columns = [self.normalize_column_name(col) for col in df.columns]
        
        # Validate required columns exist
        missing_cols = set(self.columns) - set(df.columns)
        if missing_cols:
            available_cols = ', '.join(df.columns)
            raise KeyError(f"Missing columns: {missing_cols}. Available columns: {available_cols}")
        return df

    - load_data(self, file_path: str) -> pd.DataFrame:
        try:
            
            file_loc = os.path.join(os.getcwd(), "results")
            file_loc = os.path.join(file_loc,{name})
            file_loc = os.path.join(file_loc,{llm})
            file_loc = os.path.join(file_loc,"dataset",file_path)
            
            # Detect encoding
            with open(file_path, 'rb') as file:
                raw = file.read()
                encoding = chardet.detect(raw)['encoding'] or 'utf-8'
            
            # Load with detected encoding
            df = pd.read_csv(file_path, encoding=encoding)
            
            # Validate and normalize columns
            df = self.validate_columns(df)
            
            # Convert text columns to string type
            for col in self.columns:
                df[col] = df[col].astype(str).replace('nan', '')
            
            return df
            
        except UnicodeDecodeError:
            # Fallback to latin-1
            file_loc = os.path.join(os.getcwd(), "results")
            file_loc = os.path.join(file_loc,{name})
            file_loc = os.path.join(file_loc,{llm})
            file_loc = os.path.join(file_loc,"dataset")
            file_loc = os.path.join(file_loc,file_path)
            file_path = os.path.join(file_loc, file_path)
            df = pd.read_csv(file_path, encoding='latin-1')
            df = self.validate_columns(df)
            
            for col in self.columns:
                df[col] = df[col].astype(str).replace('nan', '')
            
            return df

    - clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        return str(text).replace('\\n', ' ').replace('\\r', '')

    - tokenize(self, text: str) -> List[str]
    - remove_stopwords(self, tokens: List[str]) -> List[str]
    - lemmatize(self, tokens: List[str]) -> List[str]
    - vectorize(self, texts: List[str]) -> np.ndarray
    - transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, object]:
        # Validate columns before processing
        df = self.validate_columns(df)
        # Rest of the transform logic...

    - save_vectorizer(self, save_path: str) -> None

2. The script should:
    # Previous specifications remain the same but now with enhanced column handling
    * Write code for each function and create any additional function that is needed
    * Task to be performed: {task}
    * Process text columns: {columns}
    * Sanitize column names using normalize_column_name method
    * Get current directory using cur_direc = os.getcwd()
    * Save preprocessed data to the path. -> "cur_direc/results/{name}/{llm}/preprocessed/"
    * Save vectorizer to the path -> "cur_direc/results/{name}/{llm}/vectors/"
    * Make sure to create path for preprocessed and vectorizer
    * Include encoding="utf-8" in all file operations
    * dataset is stored in the same path by the name of test.csv

3. Required imports:
    import os
    import pandas as pd
    import numpy as np
    import chardet
    from unidecode import unidecode
    from typing import Dict, Any, List, Tuple
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import warnings
    warnings.filterwarnings('ignore')

4. Implementation requirements:
    * Use scikit-learn for vectorization (TF-IDF/CountVectorizer)
    * Use NLTK for text processing
    * Include proper error handling for encodings
    * Save preprocessing configuration
    * Handle non-ASCII characters in column names using unidecode
    * Use nltk.stem import WordNetLemmatizer for lemmatization
    * Ensure consistent column name normalization throughout the pipeline
    * Handle missing or mismatched columns gracefully with clear error messages

5. Main function should:
    * Initialize and run pipeline
    * Include encoding checks
    * Handle file encoding detection
    * Sanitize column names consistently using normalize_column_name method
    * Validate column presence before processing
    * Handle encoding-related column name mismatches

6. Constraints:
    * Do not write any comments
    * Should not include anything other than python code
    * No description should be given in the end
    * Handle both UTF-8 and non-UTF-8 encodings
    * Sanitize all text input/output
    * Ensure consistent column name handling across different encodings
    * Provide clear error messages for missing or mismatched columns

7. Run the code and make sure everything is working and return the working code only
"""
        code_files["data_preprocessing.py"] = self._generate_code(
            preprocessing_prompt, 
            columns=str(list(df.columns)), 
            task=nlp_task,
            name=self.name,
            llm=self.model
        )

        model_prompt = """
You are an ML engineer creating a training script for {task} task. Generate a Python script that implements model training pipeline.

Specifications:

1. Create a ModelTrainer class with methods:
    - init(self, config: Dict[str, Any])
    - load_preprocessed_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]
    - load_vectorizer(self, vectorizer_path: str) -> None
    - split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    - initialize_model(self) -> Any
    - train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None
    - save_model(self, save_path: str) -> None
    - save_metrics(self, metrics: Dict[str, float], save_path: str) -> None

2. The script should:
    - get curr directory from cur_direc= os.getcwd()
    - Load preprocessed data from "cur_direc/results/{name}/{llm}/preprocessed/preprocessed_data.csv"
    - Load vectorizer from 'cur_direc/results/{name}/{llm}/vectors/vectorizer.txt'
    - Split data into train/validation sets
    - Train model for task: {task}
    - Save trained model to 'cur_direc/results/{name}/{llm}/trainingmodel/'
    - Save training metrics to 'cur_direc/results/{name}/{llm}/trainingmetrics/'

3. Implementation requirements:
    - Use appropriate model for {task} task
    - Implement early stopping
    - Add model checkpointing
    - Save training configuration

4. Main function should:
    - Initialize and run training pipeline
"""


        code_files["model_training.py"] = self._generate_code(
            model_prompt,
            task=nlp_task,
            columns=str(list(df.columns)),
            name= self.name,
            llm = self.model
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