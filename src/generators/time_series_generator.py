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
        if 'timestamp' in df.columns:
            return 'regular_time_series'
        elif 'date' in df.columns and 'value' in df.columns:
            return 'financial_time_series'
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
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Handle missing values
    df.interpolate(method='time', inplace=True)
    
    # Normalization
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), 
                              columns=df.columns, 
                              index=df.index)
    
    return df_scaled, scaler

# Load and process data
df = pd.read_csv('{project_dirs["data"]}/time_series_data.csv')
processed_df, scaler = preprocess_time_series(df)
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