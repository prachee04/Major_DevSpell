import os
import ast
import numpy as np
from typing import Dict, List, Optional
import glob
from radon.raw import analyze
from radon.metrics import h_visit
from radon.complexity import cc_visit
import math

class CodePerformanceAnalysis:
    def __init__(self, project_name: str, model_name: str):
        self.project_name = project_name
        self.model_name = model_name
        self.base_path = self._construct_base_path()

    def _construct_base_path(self) -> str:
        """Constructs the correct nested path structure"""
        current_dir = os.getcwd()
        return os.path.join(
            current_dir,
            "results",
            self.project_name,
            self.model_name,
            "results"
        )

    def _get_python_files(self) -> List[str]:
        """Returns all Python files in the project directory"""
        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"Project directory not found at: {self.base_path}")

        python_files = glob.glob(os.path.join(self.base_path, "*.py"))
        if not python_files:
            raise FileNotFoundError(f"No Python files found in: {self.base_path}")

        return python_files

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate code quality metrics for all Python files in the project"""
        try:
            python_files = self._get_python_files()

            total_metrics = {
                'cyclomatic_complexity': 0,
                'maintainability_index': 0,
                'loc': 0,
                'comment_ratio': 0,
                'function_count': 0
            }

            file_count = len(python_files)

            for file_path in python_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Calculate raw metrics
                raw_metrics = analyze(content)
                total_metrics['loc'] += raw_metrics.loc

                if raw_metrics.loc > 0:  # Avoid division by zero
                    total_metrics['comment_ratio'] += (raw_metrics.comments / raw_metrics.loc) * 100

                # Calculate cyclomatic complexity
                try:
                    complexity_metrics = list(cc_visit(content))
                    if complexity_metrics:
                        total_metrics['cyclomatic_complexity'] += np.mean([m.complexity for m in complexity_metrics])
                except SyntaxError:
                    print(f"Warning: Could not parse {file_path} for complexity")

                # Calculate maintainability index
                try:
                    h_visit_result = h_visit(content)
                    hc = h_visit_result.total  # Halstead Complexity Metric
                    mi = self.calculate_maintainability_index(raw_metrics.loc, total_metrics['cyclomatic_complexity'], hc)
                    total_metrics['maintainability_index'] += mi
                except Exception as e:
                    print(f"Warning: Could not calculate maintainability index for {file_path}: {str(e)}")

                # Count functions
                try:
                    tree = ast.parse(content)
                    total_metrics['function_count'] += len([node for node in ast.walk(tree)
                                                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))])
                except SyntaxError:
                    print(f"Warning: Could not parse {file_path} for function count")

            # Calculate averages
            if file_count > 0:
                for metric in total_metrics:
                    total_metrics[metric] = round(total_metrics[metric] / file_count, 3)

            return total_metrics

        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return {
                'cyclomatic_complexity': 0.0,
                'maintainability_index': 0.0,
                'loc': 0.0,
                'comment_ratio': 0.0,
                'function_count': 0.0
            }

    def calculate_maintainability_index(self, loc: int, cc: float, hc: float) -> float:
        """Calculate Maintainability Index."""
        if loc <= 0 or hc <= 0:
            return 0.0
        try:
            mi = (171 - 5.2 * math.log(loc) - 0.23 * cc - 16.2 * math.log(hc)) * 100 / 171
            return max(0, mi)  # Ensure MI is non-negative
        except ValueError:
            return 0.0

    def analyze(self) -> Dict[str, float]:
        """Main method to run the analysis"""
        print(f"Analyzing code in directory: {self.base_path}")
        metrics = self.calculate_metrics()

        # Print the metrics in a readable format
        print("\nCode Performance Metrics:")
        print(f"Lines of Code (average): {metrics['loc']}")
        print(f"Cyclomatic Complexity (average): {metrics['cyclomatic_complexity']}")
        print(f"Maintainability Index (average): {metrics['maintainability_index']}")
        print(f"Comment Ratio % (average): {metrics['comment_ratio']}")
        print(f"Function Count (average): {metrics['function_count']}")

        return metrics

if __name__ == "__main__":
    # Example usage
    analyzer = CodePerformanceAnalysis("test", "gemma-7b-it")
    metrics = analyzer.analyze()
