import os
import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


class TimeSeriesGenerator:
    def __init__(self, groq_api_key,model):
        """Initialize with Groq API"""
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model[0],
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

    def _determine_time_series_type(self, df):
        """Determine time series analysis type based on dataset"""
        if 'timestamp' in df.columns or 'date' in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                return "multivariate"
            else:
                return "univariate"
        return "irregular"

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
        """Generate complete time series analysis project"""
        df = self._preprocess_dataset(dataset)
        project_name = f"time_series_analysis_{np.random.randint(1000, 9999)}"
        project_dirs = self._generate_project_structure(project_name)

        # Ensure the results folder exists
        results_dir = project_dirs["results"]
        os.makedirs(results_dir, exist_ok=True)

        time_series_type = self._determine_time_series_type(df)

        # Generate all code components
        code_files = {}

        # Data Preprocessing Script
        preprocessing_prompt = """
        Create a Python script for preprocessing time series data.
        Dataset columns: {columns}
        Dataset shape: {shape}
        Time Series Type: {type}
        Include:
        - Handling missing values
        - Normalization/Scaling
        - Feature engineering
        - Train-test split
        """
        code_files["data_preprocessing.py"] = self._generate_code(
            preprocessing_prompt, 
            columns=str(list(df.columns)), 
            shape=str(df.shape),
            type=time_series_type
        )

        # Forecasting Script
        forecast_prompt = """
        Create a time series forecasting script.
        Dataset columns: {columns}
        Time Series Type: {type}
        Include methods:
        - ARIMA/SARIMA
        - Prophet
        - Machine Learning models (Random Forest, XGBoost)
        """
        code_files["forecasting.py"] = self._generate_code(
            forecast_prompt,
            columns=str(list(df.columns)),
            type=time_series_type
        )

        # Evaluation Script
        eval_prompt = """
        Create a time series model evaluation script.
        Include metrics:
        - RMSE
        - MAE
        - MAPE
        - R-squared
        """
        code_files["model_evaluation.py"] = self._generate_code(eval_prompt)

        # Visualization Script
        viz_prompt = """
        Create a time series visualization script.
        Include:
        - Trend analysis
        - Seasonality decomposition
        - Correlation heatmap
        """
        code_files["visualization.py"] = self._generate_code(viz_prompt)

        # Write files to the results folder
        for filename, content in code_files.items():
            file_path = os.path.join(results_dir, filename)
            with open(file_path, "w") as f:
                f.write(content)

        # Generate project report
        report_content = f"""
# Time Series Analysis Project

## Project Overview
Generated Files:
{', '.join(code_files.keys())}

## Analysis Details
- Time Series Type: {time_series_type}
- Comprehensive preprocessing and forecasting

## Next Steps
1. Review generated scripts
2. Validate model performance
3. Tune hyperparameters
"""
        with open(os.path.join(project_dirs["docs"], "project_report.md"), "w") as f:
            f.write(report_content)

        return {
            "project_name": project_name,
            "time_series_type": time_series_type,
            "directories": project_dirs,
            "code_files": code_files,
        }