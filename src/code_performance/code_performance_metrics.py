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
            "project",
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

    def calculate_maintainability_index(self, loc: int, cc: float, hc) -> float:
        """Calculate Maintainability Index with comprehensive debugging."""
        print("\n--- Maintainability Index Calculation Debug ---")
        print(f"Input Parameters:")
        print(f"Lines of Code (LOC): {loc}")
        print(f"Cyclomatic Complexity (CC): {cc}")
        
        # Extract numeric value from HalsteadReport
        if hasattr(hc, 'difficulty'):
            # Use difficulty as a proxy for Halstead Complexity
            hc_value = hc.difficulty
        elif hasattr(hc, 'volume'):
            # Fallback to volume if difficulty is not available
            hc_value = hc.volume
        elif hasattr(hc, 'effort'):
            # Another fallback option
            hc_value = hc.effort
        else:
            # If no suitable attribute is found
            hc_value = 1.0
        
        print(f"Halstead Complexity (HC): {hc_value}")

        # Validate inputs
        if loc <= 0:
            print("ERROR: Lines of Code (LOC) is invalid (≤ 0)")
            return 0.0
        
        if cc <= 0:
            print("ERROR: Cyclomatic Complexity is invalid (≤ 0)")
            return 0.0
        
        if hc_value <= 0:
            print("ERROR: Halstead Complexity is invalid (≤ 0)")
            return 0.0
        
        try:
            # Ensure inputs are valid for logarithm
            safe_loc = max(1, abs(loc))
            safe_cc = max(1, abs(cc))
            safe_hc = max(1, abs(hc_value))

            # Calculate log values with additional logging
            log_loc = math.log(safe_loc)
            log_hc = math.log(safe_hc)

            print("\nLog Calculations:")
            print(f"Log(LOC): {log_loc}")
            print(f"Log(HC): {log_hc}")

            # Detailed MI calculation steps
            step1 = 171 - 5.2 * log_loc
            step2 = step1 - 0.23 * safe_cc
            step3 = step2 - 16.2 * log_hc
            step4 = step3 * 100 / 171

            print("\nMI Calculation Steps:")
            print(f"Step 1 (171 - 5.2 * log(LOC)): {step1}")
            print(f"Step 2 (- 0.23 * CC): {step2}")
            print(f"Step 3 (- 16.2 * log(HC)): {step3}")
            print(f"Step 4 (Normalized): {step4}")

            # Ensure MI is between 0 and 100
            mi = max(0, min(100, step4))

            print(f"\nFinal Maintainability Index: {mi}")
            return mi
        
        except Exception as e:
            print(f"CRITICAL ERROR in MI Calculation: {e}")
            print(f"Inputs - LOC: {loc}, CC: {cc}, HC: {hc_value}")
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
