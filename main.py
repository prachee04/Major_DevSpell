import os
import streamlit as st
import yaml
from dotenv import load_dotenv
import importlib
import pandas as pd
import numpy as np
import matplotlib as plt
import concurrent.futures

# Import LLMSelector and ModelEvaluator (Make sure these paths are correct based on your folder structure)
from src.utils.llm_selector import LLMSelector
from src.evaluators.model_evaluator import ModelEvaluator
from src.evaluators.recommendation_model_evaluator import RecommendationModelEvaluator

class MLProjectGenerator:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Fetch the Groq API key from environment variables
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        # Load configuration
        with open('config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize LLM Selector
        # self.llm_selector = LLMSelector(self.config['llm_providers'])
        
        # Initialize Model Evaluator
        self.model_evaluator = ModelEvaluator()

        # Initialize empty generator dictionary for lazy loading
        self.generators = {}

    def load_generators(self, llm_selector):
        """Lazy load the generators to avoid circular import issues."""
        if not self.generators:  # Only load if not already loaded
            # Dynamically import the generator classes using importlib
            RecommendationGenerator = importlib.import_module('src.generators.recommendation_generator')
            TimeSeriesGenerator = importlib.import_module('src.generators.time_series_generator')
            NLPGenerator = importlib.import_module('src.generators.nlp_generator')
            ClassificationGenerator = importlib.import_module('src.generators.classification_generator')
            # DataAnalyticsGenerator = importlib.import_module('src.generators.data_analytics_generator')
            ComputerVisionGenerator = importlib.import_module('src.generators.computer_vision_generator')

            # Initialize Generators with Groq API Key for relevant generators
            self.generators = {
                'Recommendation Systems': RecommendationGenerator.RecommendationGenerator,
                'Time Series Analysis': TimeSeriesGenerator.TimeSeriesGenerator,
                'Natural Language Processing': NLPGenerator.NLPGenerator,
                'Classification': ClassificationGenerator.ClassificationGenerator,
                # 'Data Analytics': DataAnalyticsGenerator.DataAnalyticsGenerator,
                'Computer Vision': ComputerVisionGenerator.ComputerVisionGenerator,
            }

    import concurrent.futures

    def generate_projects_in_parallel(self, generator, dataset, llms, project_type,project_name):
        """Generate projects using multiple LLMs in parallel."""
        projects = {}

        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks for each LLM
            future_to_llm = {
                executor.submit(self.process_llm, llm, generator, dataset, project_type,project_name): llm
                for llm in llms
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_llm):
                llm = future_to_llm[future]
                try:
                    # Get the result from the future
                    llm, project = future.result()
                    if project:
                        projects[llm] = project
                except Exception as e:
                    # Handle any exceptions that occurred during the processing of an LLM
                    st.error(f"Error processing LLM {llm}: {e}")

        return projects



    def run(self):
        # Load the generators dynamically
        
        # Streamlit page setup
        st.set_page_config(page_title="DevSpell AI", page_icon="🧙‍♂️")
        
        st.title("🪄 DevSpell AI: ML Project Generator")
        st.markdown("Generate end-to-end machine learning projects with AI-powered code generation.")
        
        # Sidebar for configuration
        st.sidebar.header("Project Configuration")
        project_name= st.sidebar.text_input("Enter your project name")
        selected_llms = st.sidebar.multiselect(
            "Select LLMs for Comparison", 
            self.config['llm_providers'],
        )
        if selected_llms:
            self.load_generators(selected_llms)
            
            # Project Type Selection
            project_type = st.sidebar.selectbox(
                "Select Project Domain", 
                list(self.generators.keys())  # This will now have the project types after loading generators
            )
            
            # Dataset Upload
            uploaded_dataset = st.sidebar.file_uploader(
                "Upload Dataset", 
                type=['csv', 'json', 'xlsx']
            )
            
            # Generate Projects Button
            if st.sidebar.button("Generate Projects"):
                with st.spinner("Generating ML Projects..."):
                    # Validate inputs
                    if not selected_llms:
                        st.error("Please select at least one LLM")
                        return
                    
                    # Generate Projects
                    projects = self.generate_projects(
                        project_type, 
                        uploaded_dataset, 
                        selected_llms,
                        project_name
                    )
                    
                    # Display Results
                    self.display_results(projects)

    def generate_projects(self, project_type, dataset, llms,project_name):
        generator_class = self.generators[project_type]
        projects = {}

        # Ensure dataset is provided and not empty
        if dataset is None:
            st.error("Please upload a dataset.")
            return {}

        # Generate projects in parallel, processing each LLM
        projects = self.generate_projects_in_parallel(
            generator=generator_class,
            dataset=dataset,  # Pass the raw dataset to be processed by each LLM
            llms=llms,
            project_type=project_type,
            project_name=project_name
        )

        return projects

    def process_llm(self, llm, generator_class, dataset, project_type,project_name):
        """Process a single LLM to generate the project."""
        try:
            # Initialize the LLM client using instance variables
            # llm_client = LLMSelector.get_llm(api_key=self.groq_api_key, model=llm)

            # Create the generator instance for the current LLM

            generator = generator_class(groq_api_key=self.groq_api_key, model=llm, name= project_name)

            # Generate the project and process dataset within the generator
            project = generator.generate(dataset)  # Pass the dataset for processing inside the generator
            # Ensure the project is a dictionary
            if not isinstance(project, dict):
                project = {}

            # Add additional project details
            project.update({
                'project_name': project_name,
                'recommendation_type': project_type,
                'dataset': dataset,
                'llm': llm
            })
            return llm, project
        except Exception as e:
            st.error(f"Error generating project for {llm}: {e}")
            return llm, None

    def display_results(self, projects):
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Project Details", "Model Performance", "Comparative Metrics"])
        
        with tab1:
            st.header("Generated Projects")
            for llm, project in projects.items():
                st.subheader(f"Project by {llm}")
                st.json(project)
        
        with tab2:
            st.header("Model Evaluation")
            # Iterate through projects and evaluate models
            model_metrics = {}
            for llm, project in projects.items():
                try:
                    # Ensure dataset is present and not None
                    dataset = project.get('dataset')
                    if dataset is None or (isinstance(dataset, pd.DataFrame) and dataset.empty):
                        st.warning(f"No valid dataset available for {llm}")
                        continue
                    
                    # Create a copy of the project to avoid modifying the original
                    project_copy = project.copy()
                    project_copy['dataset'] = dataset
                    
                    evaluator = RecommendationModelEvaluator(project_copy)
                    metrics = evaluator.evaluate_model(model_name='groq_llama3_70b')
                    if metrics:
                        model_metrics[llm] = metrics
                except Exception as e:
                    st.error(f"Error evaluating model for {llm}: {e}")
        
    def visualize_results(self, results):
        # Detailed visualization of results
        st.header("Model Performance Comparison")
        st.write(results)

def main():
    generator = MLProjectGenerator()
    generator.run()

if __name__ == "__main__":
    main()
