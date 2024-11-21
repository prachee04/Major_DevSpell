import os
import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class ComputerVisionGenerator:
    def __init__(self, groq_api_key):
        """Initialize with Groq API"""
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-70b-versatile",
            temperature=0.7,
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
                elif file_ext in [".xls", ".xlsx"]:
                    return pd.read_excel(dataset)
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

    def _generate_code(self, prompt_template, **kwargs):
        """Generate code using LLMChain with error handling"""
        try:
            prompt = PromptTemplate.from_template(prompt_template)
            
            input_vars = prompt.input_variables
            inputs = {}
            
            for var in input_vars:
                inputs[var] = kwargs.get(var, f"Default {var} content")
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            return chain.run(**inputs)
        
        except Exception as e:
            print(f"Error generating code: {e}")
            return f"# Error generating code\n# {str(e)}\n\n# Placeholder code"

    def _generate_project_structure(self, project_name):
        """Generate project directory structure"""
        base_dir = os.path.join(os.getcwd(), project_name)
        dirs = {
            "root": base_dir,
            "src": os.path.join(base_dir, "src"),
            "data": os.path.join(base_dir, "data"),
            "models": os.path.join(base_dir, "models"),
            "docs": os.path.join(base_dir, "docs"),
            "results": os.path.join(base_dir, "results"),
            "images": os.path.join(base_dir, "images")
        }
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        return dirs

    def _analyze_cv_dataset(self, df):
        """Analyze Computer Vision dataset characteristics"""
        # Check for image-related columns
        image_cols = [col for col in df.columns if col.lower() in ['image_path', 'image', 'filename', 'file']]
        
        # Check for label/class columns
        label_cols = [col for col in df.columns if col.lower() in ['label', 'class', 'category', 'target']]
        
        return {
            "image_columns": image_cols,
            "label_columns": label_cols,
            "total_samples": len(df),
            "cv_task": self._determine_cv_task(df, label_cols)
        }

    def _determine_cv_task(self, df, label_cols):
        """Determine the type of computer vision task"""
        if not label_cols:
            return "image_generation"
        
        unique_labels = df[label_cols[0]].nunique() if label_cols else 0
        
        if unique_labels == 2:
            return "binary_classification"
        elif unique_labels > 2:
            return "multi_class_classification"
        else:
            return "object_detection"

    def generate(self, dataset):
        """Generate complete computer vision project"""
        df = self._preprocess_dataset(dataset)
        project_name = f"computer_vision_{np.random.randint(1000, 9999)}"
        project_dirs = self._generate_project_structure(project_name)

        # Ensure the results folder exists
        results_dir = project_dirs["results"]
        os.makedirs(results_dir, exist_ok=True)

        cv_info = self._analyze_cv_dataset(df)

        # Generate all code components
        code_files = {}

        # Data Preprocessing Script
        preprocessing_prompt = """
        Create a comprehensive image preprocessing script.
        CV Task: {cv_task}
        Total Samples: {total_samples}
        Image Columns: {image_columns}
        Label Columns: {label_columns}
        Include:
        - Image augmentation
        - Resize/normalize
        - Data loading pipeline
        - Train-test split
        """
        code_files["data_preprocessing.py"] = self._generate_code(
            preprocessing_prompt, 
            cv_task=cv_info["cv_task"],
            total_samples=cv_info["total_samples"],
            image_columns=str(cv_info["image_columns"]),
            label_columns=str(cv_info["label_columns"])
        )

        # Model Architecture Script
        model_prompt = """
        Create a neural network architecture script.
        CV Task: {cv_task}
        Include architectures:
        - CNN
        - Transfer Learning (ResNet/VGG)
        - Data Augmentation strategies
        """
        code_files["model_architecture.py"] = self._generate_code(
            model_prompt,
            cv_task=cv_info["cv_task"]
        )

        # Training Script
        training_prompt = """
        Create a model training script.
        CV Task: {cv_task}
        Training techniques:
        - Learning rate scheduling
        - Early stopping
        - Model checkpointing
        """
        code_files["model_training.py"] = self._generate_code(
            training_prompt,
            cv_task=cv_info["cv_task"]
        )

        # Evaluation Script
        eval_prompt = """
        Create a model evaluation script.
        Metrics to include:
        - Accuracy
        - Precision
        - Recall
        - F1-Score
        - Confusion Matrix
        - ROC Curve
        """
        code_files["model_evaluation.py"] = self._generate_code(eval_prompt)

        # Inference Script
        inference_prompt = """
        Create an inference script for:
        - Single image prediction
        - Batch prediction
        - Model serving
        """
        code_files["inference.py"] = self._generate_code(inference_prompt)

        # Write files to the results folder
        for filename, content in code_files.items():
            file_path = os.path.join(results_dir, filename)
            with open(file_path, "w") as f:
                f.write(content)

        # Generate project report
        report_content = f"""
# Computer Vision Project

## Project Overview
Generated Files:
{', '.join(code_files.keys())}

## Dataset Characteristics
- Total Samples: {cv_info["total_samples"]}
- CV Task: {cv_info["cv_task"]}
- Image Columns: {cv_info["image_columns"]}
- Label Columns: {cv_info["label_columns"]}

## Next Steps
1. Review generated scripts
2. Validate data preprocessing
3. Train and fine-tune models
"""
        with open(os.path.join(project_dirs["docs"], "project_report.md"), "w") as f:
            f.write(report_content)

        return {
            "project_name": project_name,
            "cv_info": cv_info,
            "directories": project_dirs,
            "code_files": code_files,
        }