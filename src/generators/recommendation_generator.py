import os
import pandas as pd
import numpy as np
import io
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from src.evaluators.recommendation_model_evaluator import RecommendationModelEvaluator
import matplotlib.pyplot as plt
import seaborn as sns

class RecommendationGenerator:
    def __init__(self, groq_api_key):
        """Initialize with Groq API"""
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-70b-versatile",
            temperature=0.7,
        )

    def _create_chain(self, prompt_template):
        """Create a simple LLMChain"""
        return LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(prompt_template),
        )

    def _preprocess_dataset(self, dataset):
        """Preprocess the dataset: Either a pandas DataFrame or uploaded file"""
        # If dataset is a pandas DataFrame, return it as-is
        if isinstance(dataset, pd.DataFrame):
            return dataset

        # If dataset is a string (file path), process it
        if isinstance(dataset, str):
            if not os.path.exists(dataset):
                raise FileNotFoundError(f"Dataset file not found: {dataset}")

            file_ext = os.path.splitext(dataset)[1].lower()
            try:
                if file_ext == ".csv":
                    return pd.read_csv(dataset)
                elif file_ext == ".json":
                    return pd.read_json(dataset)
                else:
                    raise ValueError(f"Unsupported file type: {file_ext}")
            except Exception as e:
                raise ValueError(f"Error reading dataset: {str(e)}")

        # Handle file-like objects (e.g., Streamlit uploads)
        if hasattr(dataset, "name"):
            file_ext = os.path.splitext(dataset.name)[1].lower()
            try:
                if file_ext == ".csv":
                    return pd.read_csv(dataset)
                elif file_ext == ".json":
                    return pd.read_json(dataset)
                elif file_ext in [".xls", ".xlsx"]:
                    return pd.read_excel(dataset)
                else:
                    raise ValueError(f"Unsupported file type: {file_ext}")
            except Exception as e:
                raise ValueError(f"Error reading file-like object: {str(e)}")

        raise TypeError("Dataset must be a file path (CSV/JSON), DataFrame, or file-like object")

    def _generate_code(self, prompt_template, **inputs):
        """Generate code using LLMChain"""
        chain = self._create_chain(prompt_template)
        missing_vars = [var for var in chain.prompt.input_variables if var not in inputs]
        if missing_vars:
            raise ValueError(f"Missing required variables for the prompt: {missing_vars}")
        # Ensure to pass all required inputs
        return chain.run(**inputs)



    def generate(self, dataset):
        """Generate complete recommendation system project"""
        df = self._preprocess_dataset(dataset)
        project_name = f"recommender_system_{np.random.randint(1000, 9999)}"
        project_dirs = self._generate_project_structure(project_name)

        # Ensure the results folder exists
        results_dir = os.path.join(project_dirs["root"], "results")
        os.makedirs(results_dir, exist_ok=True)

        recommendation_type = self._determine_recommendation_type(df)

        # Generate all code components
        code_files = {}

        # Preprocessing Script
        preprocessing_prompt = """
        Create a Python script for preprocessing recommendation system data.
        Dataset columns: {columns}
        Dataset shape: {shape}
        Include:
        - Data cleaning
        - Feature scaling
        - Train-test split
        """
        code_files["data_preprocessing.py"] = self._generate_code(
            preprocessing_prompt, columns=list(df.columns), shape=df.shape
        )

        # Collaborative Filtering
        if recommendation_type in ["collaborative_filtering", "hybrid"]:
            collab_prompt = """
            Create a collaborative filtering recommendation system class.
            Dataset columns: {columns}
            Include:
            - User-item matrix creation
            - Similarity computation
            - Recommendation generation
            """
            code_files["collaborative_filtering.py"] = self._generate_code(
                collab_prompt, columns=list(df.columns)
            )

        # Content-Based Filtering
        if recommendation_type in ["content_based", "hybrid"]:
            content_prompt = """
            Create a content-based recommendation system class.
            Dataset columns: {columns}
            Include:
            - Feature matrix creation
            - Content similarity computation
            - Recommendation generation
            """
            code_files["content_based.py"] = self._generate_code(
                content_prompt, columns=list(df.columns)
            )

        # Evaluation Script (with corrected prompt)
        eval_prompt = """
        Create an evaluation script for recommendation systems.
        Include metrics:
        - MSE
        - Precision/Recall
        - NDCG
        - Coverage
        """
        code_files["model_evaluation.py"] = self._generate_code(eval_prompt, placeholder="example")

        # Write files to the results folder
        for filename, content in code_files.items():
            file_path = os.path.join(results_dir, filename)
            with open(file_path, "w") as f:
                f.write(content)

        # Generate project report (Optional)
        self._generate_project_report(project_dirs, code_files)

        return {
            "project_name": project_name,
            "recommendation_type": recommendation_type,
            "directories": project_dirs,
            "code_files": code_files,
        }


    def _determine_recommendation_type(self, df):
        """Determine recommendation system type based on dataset"""
        if "user_id" in df.columns and "item_id" in df.columns and "rating" in df.columns:
            return "collaborative_filtering"
        elif "content_features" in df.columns:
            return "content_based"
        else:
            return "hybrid"

    def _generate_project_structure(self, project_name):
        """Generate project directory structure"""
        base_dir = os.path.join(os.getcwd(), project_name)
        dirs = {
            "root": base_dir,
            "src": os.path.join(base_dir, "src"),
            "data": os.path.join(base_dir, "data"),
            "docs": os.path.join(base_dir, "docs"),
        }
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        return dirs

    def _generate_project_report(self, project_dirs, code_files):
        """Generate project report"""
        report_content = f"""
# Recommendation System Project

## Project Overview
Generated Files:
{', '.join(code_files.keys())}

## Implementation Details
- Dynamically generates all system components
- Includes comprehensive evaluation metrics

## Next Steps
1. Review and test generated components
2. Tune hyperparameters
3. Add additional features if required
"""
        with open(os.path.join(project_dirs["docs"], "project_report.md"), "w") as f:
            f.write(report_content)

    def generate_projects(self, project_type, dataset, llms):
        generator = self.generators[project_type]
        projects = {}

        processed_dataset = self._preprocess_dataset(dataset)
        
        for llm in llms:
            try:
                project = generator.generate(processed_dataset)
                project_details = {
                    'project_name': project['name'],
                    'dataset': processed_dataset,
                    'true_labels': project['true_labels']
                }

                # Instantiate evaluator
                evaluator = RecommendationModelEvaluator(project_details)
                metrics = evaluator.evaluate_model()

                # Ensure the metrics are being returned
                if metrics:
                    st.write(f"### Evaluation Results for {project['name']}")
                    st.write(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
                    st.write(f"Precision: {metrics['precision'] * 100:.2f}%")
                    st.write(f"Recall: {metrics['recall'] * 100:.2f}%")
                    st.write(f"F1 Score: {metrics['f1_score'] * 100:.2f}%")

                    # Visualization
                    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
                    st.write("### Model Performance Metrics")
                    st.bar_chart(metrics_df.set_index("Metric")["Value"])
                else:
                    st.write("Metrics not available.")

                # Store the project
                projects[project['name']] = project

            except Exception as e:
                st.error(f"Error generating project with LLM {llm}: {e}")

        return projects



    def plot_metrics(self, metrics):
        # Create a bar plot for the evaluation metrics
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        # Plot using seaborn or matplotlib
        plt.figure(figsize=(10, 6))
        sns.barplot(x=metric_names, y=metric_values, palette="viridis")
        plt.title("Model Evaluation Metrics")
        plt.ylabel("Score")
        plt.xlabel("Metric")
        st.pyplot(plt)