import os
import streamlit as st
import yaml
from dotenv import load_dotenv

# Import generators
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
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        with open('config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize LLM Selector
        self.llm_selector = LLMSelector(self.config['llm_providers'])
        
        # Initialize Generators
        self.generators = {
            'Recommendation Systems': RecommendationGenerator(),
            'Time Series Analysis': TimeSeriesGenerator(),
            'Natural Language Processing': NLPGenerator(),
            'Classification': ClassificationGenerator(),
            'Data Analytics': DataAnalyticsGenerator(),
            'Computer Vision': ComputerVisionGenerator()
        }
        
        # Initialize Model Evaluator
        self.model_evaluator = ModelEvaluator()
    
    def run(self):
        st.set_page_config(page_title="DevSpell AI", page_icon="🧙‍♂️")
        
        st.title("🪄 DevSpell AI: ML Project Generator")
        st.markdown("Generate end-to-end machine learning projects with AI-powered code generation.")
        
        # Sidebar for configuration
        st.sidebar.header("Project Configuration")
        
        # Project Type Selection
        project_type = st.sidebar.selectbox(
            "Select Project Domain", 
            list(self.generators.keys())
        )
        
        # Dataset Upload
        uploaded_dataset = st.sidebar.file_uploader(
            "Upload Dataset", 
            type=['csv', 'json', 'xlsx']
        )
        
        # LLM Selection
        selected_llms = st.sidebar.multiselect(
            "Select LLMs for Comparison", 
            self.config['llm_providers']
        )
        
        # Generate Projects Button
        if st.sidebar.button("Generate Projects", type="primary"):
            with st.spinner("Generating ML Projects..."):
                # Validate inputs
                if not selected_llms:
                    st.error("Please select at least one LLM")
                    return
                
                # Generate Projects
                projects = self.generate_projects(
                    project_type, 
                    uploaded_dataset, 
                    selected_llms
                )
                
                # Display Results
                self.display_results(projects)
    
    def generate_projects(self, project_type, dataset, llms):
        generator = self.generators[project_type]
        projects = {}
        
        for llm in llms:
            llm_client = self.llm_selector.get_llm(llm)
            project = generator.generate(dataset, llm_client)
            projects[llm] = project
        
        return projects
    
    def display_results(self, projects):
        # Tabs for different views
        tab1, tab2 = st.tabs(["Project Details", "Model Performance"])
        
        with tab1:
            st.header("Generated Projects")
            for llm, project in projects.items():
                st.subheader(f"Project by {llm}")
                st.json(project)
        
        with tab2:
            # Model Performance Comparison
            performance_results = self.model_evaluator.evaluate(projects)
            self.visualize_results(performance_results)
    
    def visualize_results(self, results):
        # Detailed visualization of results
        st.header("Model Performance Comparison")
        st.write(results)

def main():
    generator = MLProjectGenerator()
    generator.run()

if __name__ == "__main__":
    main()