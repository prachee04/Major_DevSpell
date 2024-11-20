import os
import numpy as np
import pandas as pd
from src.generators.base_generator import BaseGenerator

class ClassificationGenerator(BaseGenerator):
    def generate(self, dataset, llm):
        """
        Generate Classification project
        """
        df = self._preprocess_dataset(dataset)
        
        project_name = f"classification_project_{np.random.randint(1000, 9999)}"
        project_dirs = self._generate_project_structure(project_name)
        
        classification_code = self._generate_classification_code(df, project_dirs)
        self._generate_project_report(project_dirs, classification_code)
        
        return {
            'project_name': project_name,
            'project_type': 'Classification',
            'directories': project_dirs,
            'code_files': classification_code
        }
    
    def _generate_classification_code(self, df, project_dirs):
        """
        Generate classification implementation
        """
        classification_type = self._determine_classification_type(df)
        
        code_files = {
            'data_preprocessing.py': self._generate_preprocessing_script(df, project_dirs),
            'feature_engineering.py': self._generate_feature_engineering_code(),
            'model_training.py': self._generate_model_training_code(classification_type),
            'evaluation.py': self._generate_evaluation_script(classification_type)
        }
        
        for filename, code_content in code_files.items():
            file_path = os.path.join(project_dirs['src'], filename)
            with open(file_path, 'w') as f:
                f.write(code_content)
        
        return code_files
    
    def _determine_classification_type(self, df):
        """
        Determine classification task type
        """
        if len(df['target'].unique()) == 2:
            return 'binary_classification'
        elif len(df['target'].unique()) > 2:
            return 'multi_class_classification'
        else:
            return 'generic_classification'
    
    def _generate_preprocessing_script(self, df, project_dirs):
        """
        Generate data preprocessing script
        """
        preprocessing_script = f"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_classification_data(df):
    # Handle missing values
    df.dropna(inplace=True)
    
    # Encode categorical features
    le = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    
    # Scale numerical features
    scaler = StandardScaler()
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler
"""
        return preprocessing_script
    
    def _generate_feature_engineering_code(self):
        """
        Generate feature engineering utilities
        """
        return """
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

def engineer_features(X, y, k_best_features=10):
    # Feature selection using ANOVA F-value
    selector = SelectKBest(score_func=f_classif, k=k_best_features)
    X_new = selector.fit_transform(X, y)
    
    return X_new, selector
"""
    
    def _generate_model_training_code(self, classification_type):
        """
        Generate model training code
        """
        models = {
            'binary_classification': """
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_binary_classification_models(X_train, y_train):
    models = {
        'logistic_regression': LogisticRegression(),
        'random_forest': RandomForestClassifier()
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models
""",
            'multi_class_classification': """
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

def train_multi_class_models(X_train, y_train):
    models = {
        'svm': SVC(probability=True),
        'gradient_boosting': GradientBoostingClassifier()
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models
"""
        }
        return models.get(classification_type, models['multi_class_classification'])
    
    def _generate_evaluation_script(self, classification_type):
        """
        Generate model evaluation script
        """
        return """
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)

def evaluate_classification_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        
        results[name] = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1_score': f1_score(y_test, predictions, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, predictions)
        }
    
    return results
"""
    
    def _generate_project_report(self, project_dirs, code_files):
        """
        Generate project documentation
        """
        report_content = f"""
# Classification Project

## Project Overview
- Type: Machine Learning Classification
- Generated Files: {', '.join(code_files.keys())}

## Methodology
Implemented multiple classification models with feature engineering.

## Next Steps
1. Hyperparameter tuning
2. Explore ensemble methods
3. Collect more diverse training data
"""
        
        with open(os.path.join(project_dirs['docs'], 'project_report.md'), 'w') as f:
            f.write(report_content)