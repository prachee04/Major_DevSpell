import os
import numpy as np
import pandas as pd
from src.generators.base_generator import BaseGenerator

class TimeSeriesGenerator(BaseGenerator):
    def generate(self, dataset, llm):
        """
        Generate a time series analysis project
        
        Args:
            dataset (pd.DataFrame or str): Input dataset
            llm (object): Language model for code generation
        
        Returns:
            dict: Project details and generated code
        """
        # Preprocess dataset
        df = self._preprocess_dataset(dataset)
        
        # Generate project structure
        project_name = f"time_series_analysis_{np.random.randint(1000, 9999)}"
        project_dirs = self._generate_project_structure(project_name)
        
        # Generate time series analysis code
        time_series_code = self._generate_time_series_code(df, project_dirs)
        
        # Generate project report
        self._generate_project_report(project_dirs, time_series_code)
        
        return {
            'project_name': project_name,
            'project_type': 'Time Series Analysis',
            'directories': project_dirs,
            'code_files': time_series_code
        }
    
    def _preprocess_dataset(self, dataset):
        """
        Preprocess input dataset for time series analysis
        
        Args:
            dataset (pd.DataFrame or str or Streamlit UploadedFile): Input dataset
        
        Returns:
            pd.DataFrame: Preprocessed dataset
        
        Raises:
            TypeError: If dataset is not a valid type
            ValueError: If dataset cannot be processed
        """
        # Validate input type
        if dataset is None:
            raise ValueError("Dataset cannot be None")
        
        # Handle DataFrame input
        if isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
        
        # Handle Streamlit UploadedFile
        elif hasattr(dataset, 'type'):  # Streamlit UploadedFile check
            try:
                # Try reading with multiple methods based on file type
                if dataset.type == 'text/csv':
                    df = pd.read_csv(dataset)
                elif dataset.type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
                    df = pd.read_excel(dataset)
                else:
                    raise ValueError(f"Unsupported file type: {dataset.type}")
            except Exception as e:
                raise ValueError(f"Could not read Streamlit uploaded file: {e}")
        
        # Handle file path input
        elif isinstance(dataset, str):
            try:
                # Try reading with multiple methods
                try:
                    df = pd.read_csv(dataset)
                except Exception:
                    try:
                        df = pd.read_excel(dataset)
                    except Exception:
                        raise ValueError(f"Could not read dataset from {dataset}")
            except FileNotFoundError:
                raise ValueError(f"Dataset file not found: {dataset}")
        
        # Reject other input types
        else:
            raise TypeError(f"Dataset must be a file path, pandas DataFrame, or Streamlit UploadedFile. Got {type(dataset)}")
        
        # Validate DataFrame
        if df.empty:
            raise ValueError("Input dataset is empty")
        
        # Basic preprocessing steps
        # Attempt to convert timestamp/date column
        date_columns = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'timestamp'])]
        
        if date_columns:
            date_col = date_columns[0]
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df.sort_values(by=date_col, inplace=True)
            except Exception as e:
                print(f"Warning: Could not convert date column {date_col}: {e}")
        
        # Remove duplicate rows
        df.drop_duplicates(inplace=True)
        
        # Remove rows with all NaN values
        df.dropna(how='all', inplace=True)
        
        # Fill remaining NaNs with appropriate method
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].mean(), inplace=True)
        
        # Reset index after preprocessing
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def _generate_project_structure(self, project_name):
        """
        Generate project directory structure
        
        Args:
            project_name (str): Name of the project
        
        Returns:
            dict: Project directory paths
        """
        base_dir = os.path.join(os.getcwd(), project_name)
        
        # Create project directory structure
        dirs = {
            'root': base_dir,
            'src': os.path.join(base_dir, 'src'),
            'data': os.path.join(base_dir, 'data'),
            'docs': os.path.join(base_dir, 'docs'),
            'tests': os.path.join(base_dir, 'tests')
        }
        
        # Create directories
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        return dirs
    
    def _generate_time_series_code(self, df, project_dirs):
        """
        Generate time series analysis implementation
        
        Args:
            df (pd.DataFrame): Preprocessed dataset
            project_dirs (dict): Project directory paths
        
        Returns:
            dict: Generated code files
        """
        # Identify time series characteristics
        time_series_type = self._determine_time_series_type(df)
        
        # Save preprocessed dataset
        df.to_csv(os.path.join(project_dirs['data'], 'time_series_data.csv'), index=False)
        
        code_files = {
            'data_preprocessing.py': self._generate_preprocessing_script(df, project_dirs),
            'time_series_analysis.py': self._generate_analysis_code(time_series_type),
            'forecasting_model.py': self._generate_forecasting_code(time_series_type),
            'model_evaluation.py': self._generate_evaluation_script()
        }
        
        # Save code files
        for filename, code_content in code_files.items():
            file_path = os.path.join(project_dirs['src'], filename)
            with open(file_path, 'w') as f:
                f.write(code_content)
        
        return code_files
    
    def _determine_time_series_type(self, df):
        """
        Determine time series characteristics
        
        Args:
            df (pd.DataFrame): Input dataset
        
        Returns:
            str: Time series analysis type
        """
        # Basic heuristics to determine time series type
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if date_columns:
            # Check for financial indicators
            financial_indicators = ['open', 'close', 'high', 'low', 'volume', 'price']
            if any(indicator in df.columns for indicator in financial_indicators):
                return 'financial_time_series'
            
            return 'regular_time_series'
        else:
            return 'generic_time_series'
    
    def _generate_preprocessing_script(self, df, project_dirs):
        """
        Generate time series data preprocessing script
        """
        preprocessing_script = f"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_time_series(df):
    # Convert timestamp/date column
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_columns:
        date_col = date_columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    
    # Handle missing values
    df.interpolate(method='time', inplace=True)
    
    # Normalization
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df, scaler

# Load and process data
df = pd.read_csv('{project_dirs["data"]}/time_series_data.csv')
processed_df, scaler = preprocess_time_series(df)
processed_df.to_csv('{project_dirs["data"]}/processed_time_series_data.csv')
"""
        return preprocessing_script
    
    def _generate_analysis_code(self, time_series_type):
        """
        Generate time series analysis code
        """
        analysis_code = f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

class TimeSeriesAnalyzer:
    def __init__(self, df):
        self.df = df
    
    def decompose_series(self, period=12):
        # Seasonal decomposition
        decomposition = seasonal_decompose(
            self.df, 
            period=period
        )
        
        # Plotting decomposition
        plt.figure(figsize=(12,8))
        decomposition.plot()
        plt.tight_layout()
        plt.savefig('decomposition_plot.png')
    
    def detect_anomalies(self, threshold=3):
        # Detect anomalies using standard deviation
        mean = self.df.mean()
        std = self.df.std()
        anomalies = np.abs(self.df - mean) > (threshold * std)
        return self.df[anomalies]
"""
        return analysis_code
    
    def _generate_forecasting_code(self, time_series_type):
        """
        Generate time series forecasting code
        """
        forecasting_code = """
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

class TimeSeriesForecaster:
    def __init__(self, data):
        self.data = data
        self.model = None
    
    def train(self, order=(1,1,1)):
        # Split data
        train_data, test_data = train_test_split(
            self.data, 
            test_size=0.2, 
            shuffle=False
        )
        
        # Fit ARIMA model
        self.model = ARIMA(train_data, order=order)
        self.model_fit = self.model.fit()
    
    def forecast(self, steps=30):
        # Generate forecast
        forecast = self.model_fit.forecast(steps=steps)
        return forecast
"""
        return forecasting_code
    
    def _generate_evaluation_script(self):
        """
        Generate model evaluation script
        """
        evaluation_script = """
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_forecast(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    
    return {
        'mean_squared_error': mse,
        'mean_absolute_error': mae
    }
"""
        return evaluation_script
    
    def _generate_project_report(self, project_dirs, code_files):
        """
        Generate project report and documentation
        """
        report_content = f"""
# Time Series Analysis Project

## Project Overview
- Type: Time Series Analysis
- Generated Files: {', '.join(code_files.keys())}

## Methodology
Implemented time series analysis using ARIMA model with seasonal decomposition.

## Next Steps
1. Experiment with different forecasting models
2. Fine-tune model parameters
3. Collect more time-series data
"""
        
        with open(os.path.join(project_dirs['docs'], 'project_report.md'), 'w') as f:
            f.write(report_content)