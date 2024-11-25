import os
import streamlit as st
import yaml
from dotenv import load_dotenv
import importlib
import pandas as pd
import numpy as np
import matplotlib as plt
import plotly.express as px
import concurrent.futures
from projectrunner import ProjectRunnerWithErrorHandling 
from src.utils.llm_selector import LLMSelector
# from src.evaluators.model_evaluator import ModelEvaluator
from src.evaluators.recommendation_model_evaluator import RecommendationModelEvaluator
from errorhandler import LLMErrorHandler
from src.code_performance.code_performance_metrics import CodePerformanceAnalysis

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
        # self.model_evaluator = ModelEvaluator()
        
        # Initialize empty generator dictionary for lazy loading
        self.generators = {}

    def load_generators(self, llm_selector):
        """Lazy load the generators to avoid circular import issues."""
        if not self.generators:  # Only load if not already loaded
            # Dynamically import the generator classes using importlib
            
            TimeSeriesGenerator = importlib.import_module('src.generators.time_series_generator')
            NLPGenerator = importlib.import_module('src.generators.nlp_generator')
            ClassificationGenerator = importlib.import_module('src.generators.classification_generator')
            # DataAnalyticsGenerator = importlib.import_module('src.generators.data_analytics_generator')
            ComputerVisionGenerator = importlib.import_module('src.generators.computer_vision_generator')

            # Initialize Generators with Groq API Key for relevant generators
            self.generators = {

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
                    if(projects):
                        self.display_results(projects)
                        error_handler = LLMErrorHandler(selected_llms)
                        runner= ProjectRunnerWithErrorHandling(project_name, selected_llms,error_handler)
                        runner.run_all()
                    
                    # Display Results
                    

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
        tab1, tab2, tab3 = st.tabs(["Project Details", "SRS and UML", "Comparative Metrics"])
        
        with tab1:
            st.header("Generated Projects")
            for llm, project in projects.items():
                st.subheader(f"Project by {llm}")
                st.json(project)
                
        with tab2:
            st.header("Generated Documentation")
            
            # Iterate through projects
            for llm, project in projects.items():
                st.subheader(f"Documentation for {project['project_name']} by {llm}")
                
                # Check if documentation exists
                project_output_dir = f"results/{project['project_name']}/{llm}/output/"
                
                # List of documentation files to display
                doc_files = [
                    ('SRS Document', 'srs_document.md'),
                    ('Sequence Diagram', 'sequence_diagram.dot'),
                    ('Activity Diagram', 'activity_diagram.dot'),
                    ('Class Diagram', 'class_diagram.dot')
                ]
                
                # Create download buttons for each documentation file
                for file_label, filename in doc_files:
                    file_path = os.path.join(project_output_dir, filename)
                    
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_contents = f.read()
                        
                        # Download button
                        st.download_button(
                            label=f"Download {file_label}",
                            data=file_contents,
                            file_name=filename,
                            mime='text/plain' if filename.endswith('.md') else 'text/dot',
                            key=f"{llm}_{filename}"
                        )
                    else:
                        st.warning(f"{file_label} not found for {llm}")
        
        
        with tab3:
            st.header("Code Quality Metrics Across LLMs")
            
            metrics_table = []
            
            for llm, project in projects.items():
                try:
                    analyzer = CodePerformanceAnalysis(
                        project_name=project['project_name'],
                        model_name=llm
                    )
                    metrics = analyzer.analyze()
                    
                    if metrics:
                        # Add LLM name to metrics
                        metrics_dict = {
                            'LLM': llm,
                            'Lines of Code': metrics['loc'],
                            'Cyclomatic Complexity': metrics['cyclomatic_complexity'],
                            'Maintainability Index': metrics['maintainability_index'],
                            'Comment Ratio (%)': metrics['comment_ratio'],
                            'Function Count': metrics['function_count']
                        }
                        metrics_table.append(metrics_dict)
                    else:
                        st.warning(f"No code metrics available for {llm}")
                        
                except Exception as e:
                    st.error(f"Error analyzing code metrics for {llm}: {str(e)}")
                    st.error(f"Project path being checked: results/{project['project_name']}/{llm}/results/")
            
            if metrics_table:
                df_metrics = pd.DataFrame(metrics_table)
                
                # Reorder columns to put LLM first
                cols = ['LLM'] + [col for col in df_metrics.columns if col != 'LLM']
                df_metrics = df_metrics[cols]
                
                # Display metrics table
                st.subheader("Code Quality Metrics Comparison")
                st.table(df_metrics)
                
                # Create visualizations
                st.subheader("Metrics Visualization")
                
                # Bar chart for Lines of Code comparison
                loc_chart = px.bar(df_metrics, x='LLM', y='Lines of Code', 
                                title='Lines of Code Comparison',
                                color='LLM')
                st.plotly_chart(loc_chart)
                
                # Bar chart for Cyclomatic Complexity
                cc_chart = px.bar(df_metrics, x='LLM', y='Cyclomatic Complexity',
                                title='Cyclomatic Complexity Comparison',
                                color='LLM')
                st.plotly_chart(cc_chart)
                
                # Bar chart for Maintainability Index
                mi_chart = px.bar(df_metrics, x='LLM', y='Maintainability Index',
                                title='Maintainability Index Comparison',
                                color='LLM')
                st.plotly_chart(mi_chart)
                
                # Bar chart for Comment Ratio
                cr_chart = px.bar(df_metrics, x='LLM', y='Comment Ratio (%)',
                                title='Comment Ratio Comparison',
                                color='LLM')
                st.plotly_chart(cr_chart)
                
                # Bar chart for Function Count
                fc_chart = px.bar(df_metrics, x='LLM', y='Function Count',
                                title='Function Count Comparison',
                                color='LLM')
                st.plotly_chart(fc_chart)
                
                # Add interpretations
                st.subheader("Metrics Interpretation")
                st.markdown("""
                - **Lines of Code (LOC)**: Measures the size of the codebase. Lower values indicate more concise code, but should be balanced with readability.
                
                - **Cyclomatic Complexity**: Measures the number of linearly independent paths through the code. Lower values (typically <10) indicate simpler, more maintainable code.
                
                - **Maintainability Index**: Indicates how maintainable the code is on a scale of 0-100. Higher values indicate more maintainable code:
                    - 20-100: Good maintainability
                    - 10-19: Moderate maintainability
                    - 0-9: Difficult to maintain
                
                - **Comment Ratio (%)**: Percentage of comments in the code. A good balance (15-30%) suggests well-documented code without being overly verbose.
                
                - **Function Count**: Number of functions in the codebase. This indicates code modularity but should be balanced with function size and complexity.
                """)
                
                # Add overall analysis
                st.subheader("Overall Analysis")
                
                # Find best performing LLM for each metric
                best_loc = df_metrics.loc[df_metrics['Lines of Code'].idxmin()]['LLM']
                best_cc = df_metrics.loc[df_metrics['Cyclomatic Complexity'].idxmin()]['LLM']
                best_mi = df_metrics.loc[df_metrics['Maintainability Index'].idxmax()]['LLM']
                
                st.markdown(f"""
                Based on the metrics above:
                - Most concise code: **{best_loc}**
                - Least complex code: **{best_cc}**
                - Most maintainable code: **{best_mi}**
                """)
                
            else:
                st.warning("No metrics available for comparison. Please ensure code files are generated in the correct project directories.")
       

        
    def visualize_results(self, results):
        # Detailed visualization of results
        st.header("Model Performance Comparison")
        st.write(results)

def main():
    generator = MLProjectGenerator()
    generator.run()

if __name__ == "__main__":
    main()
