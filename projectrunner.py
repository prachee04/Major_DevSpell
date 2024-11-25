import os
import subprocess
import time
class ProjectRunner:
    def __init__(self, project_name, llm_used):
        self.project_name = project_name
        self.llm_used = llm_used
        self.files_to_run = {
            "data_preprocessing.py":None,
            "model_training.py": None,
        }

    def validate_files(self, llm):
        """
        Validate the existence of the required files before running them for a specific LLM.
        """
        base_dir = os.path.join("project", self.project_name, llm, "results")
        for file_name in self.files_to_run.keys():
            file_path = os.path.join(base_dir, file_name)
            if not os.path.exists(file_path):
                print(f"Error: Required file '{file_name}' is missing in '{base_dir}'")
                return False
            self.files_to_run[file_name] = file_path
        return True

    def run_script(self, script_path):
        """
        Execute a Python script and capture its output.
        """
        try:
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"Output from {script_path}:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error while executing {script_path}:\n{e.stderr}")

    def run_all(self):
        """
        Run all required scripts in sequence for each LLM.
        """
        for llm in self.llm_used:
            print(f"\nProcessing LLM: {llm}")
            if not self.validate_files(llm):
                print(f"Error: Validation failed for LLM '{llm}'. Skipping.")
                continue

            print("Running scripts in sequence...")
            for script_name, script_path in self.files_to_run.items():
                print(f"Running {script_name} for LLM '{llm}'...")
                self.run_script(script_path)
                
class ProjectRunnerWithErrorHandling(ProjectRunner):
    def __init__(self, project_name, llm_used, error_handler):
        super().__init__(project_name, llm_used)
        self.error_handler = error_handler
        
    def run_script(self, script_path: str):
        """Execute a Python script with error handling and automatic fixes."""
        retries = 0
        while retries < self.error_handler.max_retries:
            try:
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                print(f"Output from {script_path}:\n{result.stdout}")
                return True
            except subprocess.CalledProcessError as e:
                print(f"\nError occurred while executing {script_path}:")
                print(e.stderr)
                
                # Parse the error
                error_info = self.error_handler.parse_error(e.stderr)
                if not error_info["file_path"]:
                    print("Could not determine file path from error message")
                    return False
                
                # Read the current file content
                file_content = self.error_handler.read_file_content(error_info["file_path"])
                if not file_content:
                    return False
                
                # Get fix from LLM
                fixed_code = self.error_handler.get_llm_fix(error_info, file_content)
                if not fixed_code:
                    print("Could not get fix from LLM")
                    return False
                
                # Update the file with the fix
                if not self.error_handler.update_file(error_info["file_path"], fixed_code):
                    return False
                
                print(f"\nAttempted fix for {script_path}. Retrying execution...")
                retries += 1
                time.sleep(self.error_handler.retry_delay)
        
        print(f"Failed to fix {script_path} after {retries} attempts")
        return False

    def run_all(self):
        """Run all required scripts in sequence for each LLM with error handling."""
        for llm in self.llm_used:
            print(f"\nProcessing LLM: {llm}")
            if not self.validate_files(llm):
                print(f"Error: Validation failed for LLM '{llm}'. Skipping.")
                continue

            print("Running scripts in sequence...")
            for script_name, script_path in self.files_to_run.items():
                print(f"\nRunning {script_name} for LLM '{llm}'...")
                if not self.run_script(script_path):
                    print(f"Failed to successfully run {script_name} for LLM '{llm}'. Skipping remaining scripts.")
                    break
