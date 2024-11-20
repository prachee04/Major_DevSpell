import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.generators.base_generator import BaseGenerator

class RecommendationGenerator(BaseGenerator):
    def generate(self, dataset, llm):
        """
        Generate a recommendation system project
        
        Args:
            dataset (pd.DataFrame or str): Input dataset
            llm (object): Language model for code generation
        
        Returns:
            dict: Project details and generated code
        """
        # Preprocess dataset
        df = self._preprocess_dataset(dataset)
        
        # Generate project structure
        project_name = f"recommender_system_{np.random.randint(1000, 9999)}"
        project_dirs = self._generate_project_structure(project_name)
        
        # Generate recommendation system code
        recommendation_code = self._generate_recommendation_code(df, project_dirs)
        
        # Generate project report
        self._generate_project_report(project_dirs, recommendation_code)
        
        return {
            'project_name': project_name,
            'project_type': 'Recommendation System',
            'directories': project_dirs,
            'code_files': recommendation_code
        }
    
    def _generate_recommendation_code(self, df, project_dirs):
        """
        Generate recommendation system implementation
        
        Args:
            df (pd.DataFrame): Preprocessed dataset
            project_dirs (dict): Project directory paths
        
        Returns:
            dict: Generated code files
        """
        # Identify recommendation type based on dataset
        recommendation_type = self._determine_recommendation_type(df)
        
        # Generate code based on recommendation type
        code_files = {}
        
        # Data preprocessing script
        code_files['data_preprocessing.py'] = self._generate_preprocessing_script(df, project_dirs)
        
        # Model implementation scripts
        if recommendation_type == 'collaborative_filtering':
            code_files['collaborative_filtering.py'] = self._generate_collaborative_filtering_code()
        elif recommendation_type == 'content_based':
            code_files['content_based.py'] = self._generate_content_based_code()
        else:
            code_files['hybrid_recommender.py'] = self._generate_hybrid_recommender_code()
        
        # Evaluation script
        code_files['model_evaluation.py'] = self._generate_evaluation_script()
        
        # Save code files
        for filename, code_content in code_files.items():
            file_path = os.path.join(project_dirs['src'], filename)
            with open(file_path, 'w') as f:
                f.write(code_content)
        
        return code_files
    
    def _determine_recommendation_type(self, df):
        """
        Determine recommendation system type based on dataset
        
        Args:
            df (pd.DataFrame): Input dataset
        
        Returns:
            str: Recommendation system type
        """
        # Basic heuristics to determine recommendation type
        if 'user_id' in df.columns and 'item_id' in df.columns and 'rating' in df.columns:
            return 'collaborative_filtering'
        elif 'content_features' in df.columns:
            return 'content_based'
        else:
            return 'hybrid'
    
    def _generate_preprocessing_script(self, df, project_dirs):
        """
        Generate data preprocessing script
        
        Returns:
            str: Python script for data preprocessing
        """
        preprocessing_script = f"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Basic data cleaning
    df.dropna(inplace=True)
    
    # Feature scaling
    scaler = StandardScaler()
    # Add specific scaling logic based on your dataset
    
    # Train-test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    return train_df, test_df

# Load dataset
df = pd.read_csv('{project_dirs["data"]}/processed_data.csv')
train_data, test_data = preprocess_data(df)
"""
        return preprocessing_script
    
    def _generate_collaborative_filtering_code(self):
        """
        Generate collaborative filtering recommendation code
        
        Returns:
            str: Collaborative filtering implementation
        """
        collaborative_filtering_code = """
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFilteringRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.similarity_matrix = None
    
    def fit(self, train_data):
        # Create user-item matrix
        self.user_item_matrix = train_data.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        # Compute user similarity
        self.similarity_matrix = cosine_similarity(self.user_item_matrix)
    
    def recommend(self, user_id, top_n=5):
        # Find similar users
        user_index = self.user_item_matrix.index.get_loc(user_id)
        similar_users = np.argsort(self.similarity_matrix[user_index])[::-1][1:6]
        
        # Get recommendations
        user_ratings = self.user_item_matrix.iloc[user_index]
        recommendations = []
        
        return recommendations
"""
        return collaborative_filtering_code
    
    def _generate_evaluation_script(self):
        """
        Generate model evaluation script
        
        Returns:
            str: Evaluation implementation
        """
        evaluation_script = """
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support

def evaluate_recommender(model, test_data):
    # Prediction and evaluation
    predictions = model.predict(test_data)
    
    # Metrics calculation
    mse = mean_squared_error(test_data['rating'], predictions)
    
    return {
        'mean_squared_error': mse,
    }
"""
        return evaluation_script
    
    def _generate_project_report(self, project_dirs, code_files):
        """
        Generate project report and documentation
        
        Args:
            project_dirs (dict): Project directory paths
            code_files (dict): Generated code files
        """
        report_content = f"""
# Recommendation System Project

## Project Overview
- Type: Recommendation System
- Generated Files: {', '.join(code_files.keys())}

## Methodology
Implemented recommendation system using collaborative filtering approach.

## Next Steps
1. Experiment with different recommendation algorithms
2. Fine-tune hyperparameters
3. Collect more user feedback data
"""
        
        with open(os.path.join(project_dirs['docs'], 'project_report.md'), 'w') as f:
            f.write(report_content)