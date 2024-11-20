import os
import numpy as np
import pandas as pd
from src.generators.base_generator import BaseGenerator

class NLPGenerator(BaseGenerator):
    def generate(self, dataset, llm):
        """
        Generate NLP project
        """
        df = self._preprocess_dataset(dataset)
        
        project_name = f"nlp_project_{np.random.randint(1000, 9999)}"
        project_dirs = self._generate_project_structure(project_name)
        
        nlp_code = self._generate_nlp_code(df, project_dirs)
        self._generate_project_report(project_dirs, nlp_code)
        
        return {
            'project_name': project_name,
            'project_type': 'Natural Language Processing',
            'directories': project_dirs,
            'code_files': nlp_code
        }
    
    def _generate_nlp_code(self, df, project_dirs):
        """
        Generate NLP implementation
        """
        nlp_type = self._determine_nlp_type(df)
        
        code_files = {
            'data_preprocessing.py': self._generate_preprocessing_script(df, project_dirs),
            'text_preprocessing.py': self._generate_text_preprocessing_code(),
            'model_training.py': self._generate_model_training_code(nlp_type),
            'evaluation.py': self._generate_evaluation_script(nlp_type)
        }
        
        for filename, code_content in code_files.items():
            file_path = os.path.join(project_dirs['src'], filename)
            with open(file_path, 'w') as f:
                f.write(code_content)
        
        return code_files
    
    def _determine_nlp_type(self, df):
        """
        Determine NLP task type
        """
        columns = df.columns
        if 'sentiment' in columns:
            return 'sentiment_analysis'
        elif 'category' in columns:
            return 'text_classification'
        elif 'translation' in columns:
            return 'machine_translation'
        else:
            return 'generic_nlp'
    
    def _generate_preprocessing_script(self, df, project_dirs):
        """
        Generate NLP data preprocessing script
        """
        preprocessing_script = f"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_nlp_data(df):
    # Text cleaning
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace(r'[^\w\s]', '')
    
    # Encode labels
    le = LabelEncoder()
    df['encoded_label'] = le.fit_transform(df['label'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], 
        df['encoded_label'], 
        test_size=0.2, 
        random_state=42
    )
    
    return X_train, X_test, y_train, y_test, le

# Load dataset
df = pd.read_csv('{project_dirs["data"]}/nlp_data.csv')
"""
        return preprocessing_script
    
    def _generate_text_preprocessing_code(self):
        """
        Generate text preprocessing utilities
        """
        return """
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    def __init__(self, language='english'):
        nltk.download('stopwords')
        nltk.download('punkt')
        self.stop_words = set(stopwords.words(language))
    
    def preprocess_text(self, text):
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and punctuation
        tokens = [
            token for token in tokens 
            if token.isalnum() and token not in self.stop_words
        ]
        
        return ' '.join(tokens)
"""
    
    def _generate_model_training_code(self, nlp_type):
        """
        Generate model training code
        """
        return f"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def train_{nlp_type}_model(X_train, y_train):
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', MultinomialNB())
    ])
    
    model.fit(X_train, y_train)
    return model
"""
    
    def _generate_evaluation_script(self, nlp_type):
        """
        Generate NLP model evaluation script
        """
        return """
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score
)

def evaluate_nlp_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, predictions),
        'classification_report': classification_report(y_test, predictions)
    }
"""
    
    def _generate_project_report(self, project_dirs, code_files):
        """
        Generate project documentation
        """
        report_content = f"""
# NLP Project

## Project Overview
- Type: Natural Language Processing
- Generated Files: {', '.join(code_files.keys())}

## Methodology
Implemented NLP project with text preprocessing and classification.

## Next Steps
1. Experiment with advanced NLP models
2. Fine-tune preprocessing techniques
3. Explore transfer learning
"""
        
        with open(os.path.join(project_dirs['docs'], 'project_report.md'), 'w') as f:
            f.write(report_content)