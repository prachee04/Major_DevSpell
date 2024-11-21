import os
import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class ClassificationGenerator:
    def __init__(self, groq_api_key):
        """Initialize with Groq API"""
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-70b-versatile",
            temperature=0.7,
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
            prompt = PromptTemplate.from_template(prompt_template)
            
            input_vars = prompt.input_variables
            inputs = {}
            
            for var in input_vars:
                inputs[var] = kwargs.get(var, f"Default {var} content")
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            return chain.run(**inputs)
        
        except Exception as e:
            print(f"Error generating code: {e}")
            return f"# Error generating code\n# {str(e)}\n\n# Placeholder code"

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

    def _analyze_classification_problem(self, df):
        """Analyze classification characteristics"""
        # Identify target variable and classification type
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Assuming the last categorical column is the target
        if len(categorical_cols) > 0:
            target_col = categorical_cols[-1]
            classes = df[target_col].unique()
            
            return {
                "target_column": target_col,
                "num_classes": len(classes),
                "class_distribution": dict(df[target_col].value_counts()),
                "is_binary": len(classes) == 2
            }
        
        return None

    def generate(self, dataset):
        """Generate complete classification project"""
        df = self._preprocess_dataset(dataset)
        project_name = f"classification_project_{np.random.randint(1000, 9999)}"
        project_dirs = self._generate_project_structure(project_name)

        # Ensure the results folder exists
        results_dir = project_dirs["results"]
        os.makedirs(results_dir, exist_ok=True)

        classification_info = self._analyze_classification_problem(df)
        
        if classification_info is None:
            raise ValueError("No suitable target variable found for classification")

        # Generate all code components
        code_files = {}

        # Data Preprocessing Script
        preprocessing_prompt = """
        Create a Python script for classification data preprocessing.
        Target Column: {target_column}
        Number of Classes: {num_classes}
        Class Distribution: {class_distribution}
        Include:
        - Feature scaling
        - Feature selection
        - Handling missing values
        - Encoding categorical variables
        """
        code_files["data_preprocessing.py"] = self._generate_code(
            preprocessing_prompt, 
            target_column=classification_info["target_column"],
            num_classes=classification_info["num_classes"],
            class_distribution=str(classification_info["class_distribution"])
        )

        # Model Training Script
        model_prompt = """
        Create a classification model training script.
        Target: {target_column}
        Classification Type: {classification_type}
        Include models:
        - Logistic Regression
        - Random Forest
        - Support Vector Machine
        - Gradient Boosting
        """
        classification_type = "Binary" if classification_info["is_binary"] else "Multi-class"
        code_files["model_training.py"] = self._generate_code(
            model_prompt,
            target_column=classification_info["target_column"],
            classification_type=classification_type
        )

        # Model Evaluation Script
        eval_prompt = """
        Create a classification model evaluation script.
        Metrics to include:
        - Accuracy
        - Precision
        - Recall
        - F1-Score
        - Confusion Matrix
        - ROC AUC Curve
        """
        code_files["model_evaluation.py"] = self._generate_code(eval_prompt)

        # Feature Importance Script
        feature_prompt = """
        Create a feature importance analysis script.
        Techniques:
        - Correlation analysis
        - Mutual information
        - Feature importance from tree-based models
        """
        code_files["feature_importance.py"] = self._generate_code(feature_prompt)

        # Write files to the results folder
        for filename, content in code_files.items():
            file_path = os.path.join(results_dir, filename)
            with open(file_path, "w") as f:
                f.write(content)

        # Generate project report
        report_content = f"""
# Classification Project

## Project Overview
Generated Files:
{', '.join(code_files.keys())}

## Classification Problem Details
- Target Column: {classification_info["target_column"]}
- Number of Classes: {classification_info["num_classes"]}
- Classification Type: {classification_type}

## Class Distribution
{classification_info["class_distribution"]}

## Next Steps
1. Review generated scripts
2. Validate preprocessing
3. Tune model hyperparameters
"""
        with open(os.path.join(project_dirs["docs"], "project_report.md"), "w") as f:
            f.write(report_content)

        return {
            "project_name": project_name,
            "classification_info": classification_info,
            "directories": project_dirs,
             "code_files": code_files,
        }