import os
import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class DataAnalyticsGenerator:
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

    def _determine_dataset_type(self, df):
        """Determine dataset type based on columns and data"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        return {
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "total_columns": len(df.columns)
        }

    def generate(self, dataset):
        """Generate complete data analytics project"""
        df = self._preprocess_dataset(dataset)
        project_name = f"data_analytics_{np.random.randint(1000, 9999)}"
        project_dirs = self._generate_project_structure(project_name)

        # Ensure the results folder exists
        results_dir = project_dirs["results"]
        os.makedirs(results_dir, exist_ok=True)

        dataset_type = self._determine_dataset_type(df)

        # Generate all code components
        code_files = {}

        # Data Cleaning Script
        cleaning_prompt = """
        Create a Python script for data cleaning and preprocessing.
        Dataset columns: {columns}
        Dataset shape: {shape}
        Numeric Columns: {numeric_cols}
        Categorical Columns: {categorical_cols}
        Include:
        - Handling missing values
        - Outlier detection and treatment
        - Feature encoding
        - Data validation
        """
        code_files["data_cleaning.py"] = self._generate_code(
            cleaning_prompt, 
            columns=str(list(df.columns)), 
            shape=str(df.shape),
            numeric_cols=dataset_type["numeric_columns"],
            categorical_cols=dataset_type["categorical_columns"]
        )

        # Exploratory Data Analysis Script
        eda_prompt = """
        Create an EDA script with:
        - Descriptive statistics
        - Distribution analysis
        - Correlation matrix
        - Visualization of key insights
        Dataset columns: {columns}
        """
        code_files["exploratory_analysis.py"] = self._generate_code(
            eda_prompt,
            columns=str(list(df.columns))
        )

        # Statistical Analysis Script
        stats_prompt = """
        Create a statistical analysis script.
        Include:
        - Hypothesis testing
        - Confidence intervals
        - Key statistical measures
        """
        code_files["statistical_analysis.py"] = self._generate_code(stats_prompt)

        # Visualization Script
        viz_prompt = """
        Create a comprehensive visualization script.
        Include:
        - Box plots
        - Scatter plots
        - Pair plots
        - Distribution plots
        """
        code_files["visualization.py"] = self._generate_code(viz_prompt)

        # Write files to the results folder
        for filename, content in code_files.items():
            file_path = os.path.join(results_dir, filename)
            with open(file_path, "w") as f:
                f.write(content)

        # Generate project report
        report_content = f"""
# Data Analytics Project

## Project Overview
Generated Files:
{', '.join(code_files.keys())}

## Dataset Characteristics
- Total Columns: {dataset_type["total_columns"]}
- Numeric Columns: {dataset_type["numeric_columns"]}
- Categorical Columns: {dataset_type["categorical_columns"]}

## Next Steps
1. Review generated scripts
2. Validate data cleaning
3. Interpret statistical findings
"""
        with open(os.path.join(project_dirs["docs"], "project_report.md"), "w") as f:
            f.write(report_content)

        return {
            "project_name": project_name,
            "dataset_type": dataset_type,
            "directories": project_dirs,
            "code_files": code_files,
        }