import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, precision_score, recall_score, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class RecommendationModelEvaluator:
    def __init__(self, project_details):
        self.project_name = project_details.get('project_name', 'Unnamed Project')
        self.dataset = project_details.get('dataset')

        # Check dataset validity
        if self.dataset is None or self.dataset.empty:
            st.error("No dataset provided or dataset is empty.")
            raise ValueError("Dataset cannot be empty.")

        st.write("Dataset loaded in model evaluator:", self.dataset.head())

    def _auto_select_target(self):
        numeric_columns = self.dataset.select_dtypes(include=[np.number]).columns
        st.write("Available numeric columns:", numeric_columns)

        if numeric_columns.empty:
            raise ValueError("Dataset contains no numeric columns for target selection.")

        # Try priority-based selection
        for candidate in ['rating', 'score', 'value', 'target']:
            if candidate in numeric_columns:
                return candidate

        # Default to the last numeric column
        return numeric_columns[-1]

    def prepare_data(self, target_column=None):
        st.write("Preparing dataset for training...")

        if self.dataset.empty:
            raise ValueError("Dataset is empty. Cannot prepare data.")

        # Check for valid numeric columns
        numeric_columns = self.dataset.select_dtypes(include=[np.number]).columns
        if numeric_columns.empty:
            raise ValueError("No numeric columns found in the dataset.")

        # Auto-select target if not provided
        if target_column is None:
            target_column = self._auto_select_target()
        st.write("Selected target column:", target_column)

        # Ensure target column exists
        if target_column not in self.dataset.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        # Split features and target
        X = self.dataset.drop(columns=[target_column])
        y = self.dataset[target_column]

        # Encode categorical columns
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        st.write("Data prepared successfully. Features scaled.")

        # Train-test split
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def evaluate_model(self):
        # Simulated evaluation metrics
        accuracy = 0.85
        precision = 0.8
        recall = 0.75
        f1 = 0.77

        # Debug print
        print(f"Metrics: accuracy={accuracy}, precision={precision}, recall={recall}, f1={f1}")

        # Return the metrics
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }