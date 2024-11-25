import os
import graphviz
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


class DocumentGenerator:
    def __init__(self, groq_api_key, model_name):
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=0.7
        )

    def _get_results_dir(self, project_name: str, model_name: str) -> str:
        """Construct the path to the results directory."""
        current_dir = os.getcwd()
        return os.path.join(current_dir, "project", project_name, model_name, "results")

    def _read_python_files(self, results_dir: str) -> str:
        """Read all Python files in the results directory and return their combined content."""
        code_content = []
        try:
            for file_name in os.listdir(results_dir):
                if file_name.endswith(".py"):
                    file_path = os.path.join(results_dir, file_name)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_content.append(f.read())
        except Exception as e:
            print(f"Error reading Python files: {e}")
        return "\n".join(code_content)

    def generate_srs_document(self, project_details: Dict[str, Any], code_content: str) -> str:
        """Generate Software Requirements Specification document."""
        print("Generating SRS Document...")
        srs_prompt = PromptTemplate.from_template("""
        Create a comprehensive Software Requirements Specification (SRS) document for a project with the following details:
        - Project Name: {project_name}
        - NLP Task: {nlp_task}
        - Dataset Columns: {columns}
        - Codebase Summary: {code_summary}

        Sections to include:
        1. Introduction
        2. Overall Description
        3. System Features
        4. External Interface Requirements
        5. Non-Functional Requirements
        6. Constraints
        7. Preliminary Design
        """)

        chain = LLMChain(llm=self.llm, prompt=srs_prompt)
        srs_content = chain.run({
            'project_name': project_details.get('project_name', 'Undefined Project'),
            'nlp_task': project_details.get('nlp_task', 'Undefined NLP Task'),
            'columns': str(project_details.get('columns', [])),
            'code_summary': code_content[:1000]  # Use a truncated summary if necessary
        })
        return srs_content

    def generate_sequence_diagram(self, project_details: Dict[str, Any]) -> str:
        """Generate a sequence diagram."""
        print("Generating Sequence Diagram...")
        return f"""
        digraph sequence_diagram {{
            rankdir=TB;
            node [shape=box, style=rounded];
            
            User [label="User"];
            System [label="System"];
            Database [label="Database"];
            
            User -> System [label="Request"];
            System -> Database [label="Query"];
            Database -> System [label="Response"];
            System -> User [label="Result"];
        }}
        """

    def generate_activity_diagram(self, project_details: Dict[str, Any]) -> str:
        """Generate an activity diagram."""
        print("Generating Activity Diagram...")
        return f"""
        digraph activity_diagram {{
            rankdir=TB;
            node [shape=box, style=rounded];
            start [shape=circle, label="Start"];
            end [shape=doublecircle, label="End"];
            
            LoadData [label="Load Data"];
            PreprocessData [label="Preprocess Data"];
            TrainModel [label="Train Model"];
            EvaluateModel [label="Evaluate Model"];
            DeployModel [label="Deploy Model"];
            
            start -> LoadData;
            LoadData -> PreprocessData;
            PreprocessData -> TrainModel;
            TrainModel -> EvaluateModel;
            EvaluateModel -> DeployModel;
            DeployModel -> end;
        }}
        """

    def generate_class_diagram(self, project_details: Dict[str, Any]) -> str:
        """Generate a class diagram."""
        print("Generating Class Diagram...")
        return f"""
        digraph class_diagram {{
            rankdir=TB;
            node [shape=record];
            
            DataLoader [label="{{DataLoader|+ load_data()}}"];
            Preprocessor [label="{{Preprocessor|+ preprocess()}}"];
            ModelTrainer [label="{{ModelTrainer|+ train()}}"];
            
            DataLoader -> Preprocessor [label="provides data"];
            Preprocessor -> ModelTrainer [label="trains model"];
        }}
        """

    def generate_documentation(self, project_details: Dict[str, Any]):
        """Generate all documentation files dynamically from the project directory."""
        project_name = project_details.get('project_name', 'Undefined_Project')
        model_name = project_details.get('model_name', 'Undefined_Model')

        # Define the result directory
        result_files_dir = self._get_results_dir(project_name, model_name)
        os.makedirs(result_files_dir, exist_ok=True)  # Ensure the directory exists

        print(f"Generating documentation for project: {project_name}, model: {model_name}")
        print(f"Results directory: {result_files_dir}")

        try:
            # Generate content
            code_content = self._read_python_files(result_files_dir)
            srs_doc = self.generate_srs_document(project_details, code_content)
            sequence_diagram = self.generate_sequence_diagram(project_details)
            activity_diagram = self.generate_activity_diagram(project_details)
            class_diagram = self.generate_class_diagram(project_details)

            # File paths
            output_files = {
                'srs_document': os.path.join(result_files_dir, 'srs_document.md'),
                'sequence_diagram': os.path.join(result_files_dir, 'sequence_diagram.dot'),
                'activity_diagram': os.path.join(result_files_dir, 'activity_diagram.dot'),
                'class_diagram': os.path.join(result_files_dir, 'class_diagram.dot')
            }

            # Write content to files
            with open(output_files['srs_document'], 'w', encoding='utf-8') as f:
                print(f"Writing SRS Document to {output_files['srs_document']}")
                f.write(srs_doc)
            with open(output_files['sequence_diagram'], 'w', encoding='utf-8') as f:
                f.write(sequence_diagram)
            with open(output_files['activity_diagram'], 'w', encoding='utf-8') as f:
                f.write(activity_diagram)
            with open(output_files['class_diagram'], 'w', encoding='utf-8') as f:
                f.write(class_diagram)

            print(f"Documents successfully generated in {result_files_dir}")
            return output_files

        except Exception as e:
            print(f"Error generating documentation: {e}")
            return {}
