import os
import streamlit as st
import yaml
from dotenv import load_dotenv
import importlib

# Import LLMSelector and ModelEvaluator (Make sure these paths are correct based on your folder structure)
from src.utils.llm_selector import LLMSelector
from src.evaluators.model_evaluator import ModelEvaluator

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
        self.llm_selector = LLMSelector(self.config['llm_providers'])
        
        # Initialize Model Evaluator
        self.model_evaluator = ModelEvaluator()

        # Initialize empty generator dictionary for lazy loading
        self.generators = {}

    def load_generators(self):
        """Lazy load the generators to avoid circular import issues."""
        if not self.generators:  # Only load if not already loaded
            # Dynamically import the generator classes using importlib
            RecommendationGenerator = importlib.import_module('src.generators.recommendation_generator')
            TimeSeriesGenerator = importlib.import_module('src.generators.time_series_generator')
            NLPGenerator = importlib.import_module('src.generators.nlp_generator')
            ClassificationGenerator = importlib.import_module('src.generators.classification_generator')
            DataAnalyticsGenerator = importlib.import_module('src.generators.data_analytics_generator')
            ComputerVisionGenerator = importlib.import_module('src.generators.computer_vision_generator')

            # Initialize Generators with Groq API Key for relevant generators
            self.generators = {
                'Recommendation Systems': RecommendationGenerator.RecommendationGenerator(groq_api_key=self.groq_api_key),
                'Time Series Analysis': TimeSeriesGenerator.TimeSeriesGenerator(groq_api_key=self.groq_api_key),
                'Natural Language Processing': NLPGenerator.NLPGenerator(groq_api_key=self.groq_api_key),
                'Classification': ClassificationGenerator.ClassificationGenerator(groq_api_key=self.groq_api_key),
                'Data Analytics': DataAnalyticsGenerator.DataAnalyticsGenerator(groq_api_key=self.groq_api_key),
                'Computer Vision': ComputerVisionGenerator.ComputerVisionGenerator(groq_api_key=self.groq_api_key)
            }

    def run(self):
        # Load the generators dynamically
        self.load_generators()

        # Streamlit page setup
        st.set_page_config(page_title="DevSpell AI", page_icon="🧙‍♂️")
        
        st.title("🪄 DevSpell AI: ML Project Generator")
        st.markdown("Generate end-to-end machine learning projects with AI-powered code generation.")
        
        # Sidebar for configuration
        st.sidebar.header("Project Configuration")
        
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
        
        # LLM Selection
        selected_llms = st.sidebar.multiselect(
            "Select LLMs for Comparison", 
            self.config['llm_providers']
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
                    selected_llms
                )
                
                # Display Results
                self.display_results(projects)
    
    def generate_projects(self, project_type, dataset, llms):
        # Get the appropriate generator based on project type
        generator = self.generators[project_type]
        projects = {}
        
        # Generate projects for each selected LLM
        for llm in llms:
            llm_client = self.llm_selector.get_llm(llm)
            project = generator.generate(dataset)  # Pass both dataset and llm_client
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
