import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.generators.base_generator import BaseGenerator

class DataAnalyticsGenerator(BaseGenerator):
    def generate(self, dataset, llm):
        """
        Generate a data analytics project
        
        Args:
            dataset (pd.DataFrame or str): Input dataset
            llm (object): Language model for code generation
        
        Returns:
            dict: Project details and generated code
        """
        # Preprocess dataset
        df = self._preprocess_dataset(dataset)
        
        # Generate project structure
        project_name = f"data_analytics_{np.random.randint(1000, 9999)}"
        project_dirs = self._generate_project_structure(project_name)
        
        # Generate data analytics code
        analytics_code = self._generate_analytics_code(df, project_dirs)
        
        # Generate project report
        self._generate_project_report(project_dirs, analytics_code)
        
        return {
            'project_name': project_name,
            'project_type': 'Data Analytics',
            'directories': project_dirs,
            'code_files': analytics_code
        }
    
    def _generate_analytics_code(self, df, project_dirs):
        """
        Generate data analytics implementation
        
        Args:
            df (pd.DataFrame): Preprocessed dataset
            project_dirs (dict): Project directory paths
        
        Returns:
            dict: Generated code files
        """
        code_files = {}
        
        # Data preprocessing script
        code_files['data_preprocessing.py'] = self._generate_preprocessing_script(df, project_dirs)
        
        # Exploratory data analysis script
        code_files['exploratory_analysis.py'] = self._generate_eda_script(df)
        
        # Statistical analysis script
        code_files['statistical_analysis.py'] = self._generate_statistical_analysis_script(df)
        
        # Visualization script
        code_files['data_visualization.py'] = self._generate_visualization_script(df)
        
        # Save code files
        for filename, code_content in code_files.items():
            file_path = os.path.join(project_dirs['src'], filename)
            with open(file_path, 'w') as f:
                f.write(code_content)
        
        return code_files
    
    def _generate_preprocessing_script(self, df, project_dirs):
        """
        Generate data preprocessing script
        
        Returns:
            str: Python script for data preprocessing
        """
        preprocessing_script = f"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    # Identify numeric and categorical columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit and transform the data
    processed_data = preprocessor.fit_transform(df)
    
    return processed_data, preprocessor

# Load dataset
df = pd.read_csv('{project_dirs["data"]}/raw_data.csv')
processed_data, preprocessor = preprocess_data(df)
"""
        return preprocessing_script
    
    def _generate_eda_script(self, df):
        """
        Generate exploratory data analysis script
        
        Returns:
            str: Exploratory data analysis implementation
        """
        eda_script = """
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_exploratory_analysis(df):
    # Basic dataset information
    print(df.info())
    
    # Summary statistics
    print(df.describe())
    
    # Check for missing values
    print(df.isnull().sum())
    
    # Correlation matrix
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

# Load dataset
df = pd.read_csv('processed_data.csv')
perform_exploratory_analysis(df)
"""
        return eda_script
    
    def _generate_statistical_analysis_script(self, df):
        """
        Generate statistical analysis script
        
        Returns:
            str: Statistical analysis implementation
        """
        statistical_script = """
import pandas as pd
import scipy.stats as stats

def perform_statistical_analysis(df):
    # T-test example (modify based on your specific dataset)
    # Assumes binary classification or group comparison
    groups = df['target'].unique()
    group1 = df[df['target'] == groups[0]]['feature']
    group2 = df[df['target'] == groups[1]]['feature']
    
    t_statistic, p_value = stats.ttest_ind(group1, group2)
    
    print(f"T-test results:")
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")
    
    # Additional statistical tests can be added here

# Load dataset
df = pd.read_csv('processed_data.csv')
perform_statistical_analysis(df)
"""
        return statistical_script
    
    def _generate_visualization_script(self, df):
        """
        Generate data visualization script
        
        Returns:
            str: Visualization implementation
        """
        visualization_script = """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_visualizations(df):
    # Histogram of numeric features
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_features].hist(figsize=(12, 8))
    plt.tight_layout()
    plt.savefig('numeric_features_histogram.png')
    plt.close()
    
    # Box plot for feature distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='target', y='feature')
    plt.title('Feature Distribution by Target')
    plt.tight_layout()
    plt.savefig('boxplot_by_target.png')
    plt.close()

# Load dataset
df = pd.read_csv('processed_data.csv')
create_visualizations(df)
"""
        return visualization_script
    
    def _generate_project_report(self, project_dirs, code_files):
        """
        Generate project report and documentation
        
        Args:
            project_dirs (dict): Project directory paths
            code_files (dict): Generated code files
        """
        report_content = f"""
# Data Analytics Project

## Project Overview
- Type: Data Analytics
- Generated Files: {', '.join(code_files.keys())}

## Methodology
Performed comprehensive data analysis including:
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Statistical Analysis
- Data Visualization

## Next Steps
1. Validate and interpret statistical findings
2. Develop additional visualizations
3. Consider advanced machine learning modeling
"""
        
        with open(os.path.join(project_dirs['docs'], 'project_report.md'), 'w') as f:
            f.write(report_content)