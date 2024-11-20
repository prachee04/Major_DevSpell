import streamlit as st
import yaml
from src.generators import (
    RecommendationGenerator,
    TimeSeriesGenerator,
    NLPGenerator,
    ClassificationGenerator,
    DataAnalyticsGenerator,
    ComputerVisionGenerator
)
from src.evaluators.model_evaluator import ModelEvaluator
from src.utils.llm_selector import LLMSelector

class MLProjectGenerator:
    def __init__(self):
        # Load configuration
        with open('config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize LLM Selector
        self.llm_selector = LLMSelector(self.config['llm_providers'])
        
        # Initialize Generators
        self.generators = {
            'recommendation_systems': RecommendationGenerator(),
            'time_series_analysis': TimeSeriesGenerator(),
            'nlp': NLPGenerator(),
            'classification': ClassificationGenerator(),
            'data_analytics': DataAnalyticsGenerator(),
            'computer_vision': ComputerVisionGenerator()
        }
        
        # Initialize Model Evaluator
        self.model_evaluator = ModelEvaluator()
    
    def run(self):
        st.title("ML Project Generator")
        
        # Project Type Selection
        project_type = st.selectbox(
            "Select Project Type", 
            list(self.generators.keys())
        )
        
        # Dataset Upload
        uploaded_dataset = st.file_uploader(
            "Upload Dataset", 
            type=['csv', 'json', 'xlsx']
        )
        
        # LLM Selection
        selected_llms = st.multiselect(
            "Select LLMs for Comparison", 
            self.config['llm_providers']
        )
        
        if st.button("Generate Projects"):
            # Generate Projects using selected LLMs
            projects = self.generate_projects(
                project_type, 
                uploaded_dataset, 
                selected_llms
            )
            
            # Evaluate and Compare Projects
            self.compare_projects(projects)
    
    def generate_projects(self, project_type, dataset, llms):
        generator = self.generators[project_type]
        projects = {}
        
        for llm in llms:
            project = generator.generate(
                dataset, 
                self.llm_selector.get_llm(llm)
            )
            projects[llm] = project
        
        return projects
    
    def compare_projects(self, projects):
        # Model Performance Comparison
        performance_results = self.model_evaluator.evaluate(projects)
        
        # Visualization of Results
        self.visualize_results(performance_results)
    
    def visualize_results(self, results):
        # Create comparative visualizations
        pass

def main():
    generator = MLProjectGenerator()
    generator.run()

if __name__ == "__main__":
    main()