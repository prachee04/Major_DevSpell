import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.generators.base_generator import BaseGenerator

class ComputerVisionGenerator(BaseGenerator):
    def generate(self, dataset, llm):
        """
        Generate a computer vision project
        
        Args:
            dataset (pd.DataFrame or str): Input dataset
            llm (object): Language model for code generation
        
        Returns:
            dict: Project details and generated code
        """
        # Preprocess dataset
        df = self._preprocess_dataset(dataset)
        
        # Generate project structure
        project_name = f"computer_vision_{np.random.randint(1000, 9999)}"
        project_dirs = self._generate_project_structure(project_name)
        
        # Generate computer vision project code
        cv_code = self._generate_computer_vision_code(df, project_dirs)
        
        # Generate project report
        self._generate_project_report(project_dirs, cv_code)
        
        return {
            'project_name': project_name,
            'project_type': 'Computer Vision',
            'directories': project_dirs,
            'code_files': cv_code
        }
    
    def _preprocess_dataset(self, dataset):
        """
        Preprocess input dataset for computer vision task
        
        Args:
            dataset (pd.DataFrame or str): Input dataset
        
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        if isinstance(dataset, str):
            df = pd.read_csv(dataset)
        elif isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
        else:
            raise ValueError("Dataset must be a file path or pandas DataFrame")
        
        return df
    
    def _generate_project_structure(self, project_name):
        """
        Create project directory structure
        
        Args:
            project_name (str): Name of the project
        
        Returns:
            dict: Project directory paths
        """
        base_dir = os.path.join('projects', project_name)
        
        project_dirs = {
            'root': base_dir,
            'src': os.path.join(base_dir, 'src'),
            'data': os.path.join(base_dir, 'data'),
            'models': os.path.join(base_dir, 'models'),
            'docs': os.path.join(base_dir, 'docs'),
            'tests': os.path.join(base_dir, 'tests')
        }
        
        # Create directories
        for dir_path in project_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        return project_dirs
    
    def _generate_computer_vision_code(self, df, project_dirs):
        """
        Generate computer vision project implementation
        
        Args:
            df (pd.DataFrame): Preprocessed dataset
            project_dirs (dict): Project directory paths
        
        Returns:
            dict: Generated code files
        """
        # Determine computer vision task type
        cv_task = self._determine_computer_vision_task(df)
        
        # Generate code files
        code_files = {}
        
        # Data preprocessing script
        code_files['data_preprocessing.py'] = self._generate_preprocessing_script(df, project_dirs)
        
        # Model implementation scripts based on task
        if cv_task == 'classification':
            code_files['image_classifier.py'] = self._generate_classification_code()
        elif cv_task == 'object_detection':
            code_files['object_detector.py'] = self._generate_object_detection_code()
        elif cv_task == 'segmentation':
            code_files['image_segmentation.py'] = self._generate_segmentation_code()
        
        # Evaluation script
        code_files['model_evaluation.py'] = self._generate_evaluation_script()
        
        # Save code files
        for filename, code_content in code_files.items():
            file_path = os.path.join(project_dirs['src'], filename)
            with open(file_path, 'w') as f:
                f.write(code_content)
        
        return code_files
    
    def _determine_computer_vision_task(self, df):
        """
        Determine computer vision task type based on dataset
        
        Args:
            df (pd.DataFrame): Input dataset
        
        Returns:
            str: Computer vision task type
        """
        # Basic heuristics to determine CV task
        if 'label' in df.columns and 'image_path' in df.columns:
            return 'classification'
        elif 'bounding_box' in df.columns:
            return 'object_detection'
        elif 'segmentation_mask' in df.columns:
            return 'segmentation'
        else:
            return 'classification'  # Default to classification
    
    def _generate_preprocessing_script(self, df, project_dirs):
        """
        Generate data preprocessing script for computer vision
        
        Returns:
            str: Python script for data preprocessing
        """
        preprocessing_script = f"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_images(df, img_size=(224, 224)):
    # Image data generator for augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    return train_df, test_df, datagen

# Load dataset
df = pd.read_csv('{project_dirs["data"]}/image_metadata.csv')
train_data, test_data, image_generator = preprocess_images(df)
"""
        return preprocessing_script
    
    def _generate_classification_code(self):
        """
        Generate image classification code
        
        Returns:
            str: Image classification implementation
        """
        classification_code = """
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class ImageClassifier:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self):
        # Pre-trained ResNet50 as base
        base_model = ResNet50(
            weights='imagenet', 
            include_top=False, 
            input_shape=(224, 224, 3)
        )
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Add classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_generator, validation_generator, epochs=10):
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs
        )
        return history
"""
        return classification_code
    
    def _generate_evaluation_script(self):
        """
        Generate model evaluation script
        
        Returns:
            str: Evaluation implementation
        """
        evaluation_script = """
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, test_generator):
    # Predict on test data
    predictions = model.predict(test_generator)
    true_labels = test_generator.classes
    
    # Classification report
    class_report = classification_report(
        true_labels, 
        predictions.argmax(axis=1)
    )
    
    # Confusion matrix visualization
    cm = confusion_matrix(true_labels, predictions.argmax(axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    return class_report
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
# Computer Vision Project

## Project Overview
- Type: Computer Vision
- Generated Files: {', '.join(code_files.keys())}

## Methodology
Implemented computer vision project focusing on image classification/object detection.

## Next Steps
1. Experiment with different neural network architectures
2. Fine-tune hyperparameters
3. Collect more diverse training data
"""
        
        with open(os.path.join(project_dirs['docs'], 'project_report.md'), 'w') as f:
            f.write(report_content)