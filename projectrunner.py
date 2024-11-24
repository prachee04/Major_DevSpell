import os
import subprocess

class ProjectRunner:
    def __init__(self, project_name, llm_used):
        self.base_dir = os.path.join("results", project_name, llm_used[0], "results")
        self.files_to_run = {
            "model_training.py": None,
        }

    def validate_files(self):
        """
        Validate the existence of the required files before running them.
        """
        for file_name in self.files_to_run.keys():
            file_path = os.path.join(self.base_dir, file_name)
            if not os.path.exists(file_path):
                print(f"Error: Required file '{file_name}' is missing in '{self.base_dir}'")
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
        Run all required scripts in sequence.
        """
        if not self.validate_files():
            print("Error: Validation failed. Exiting.")
            return

        print("Running scripts in sequence...")
        for script_name, script_path in self.files_to_run.items():
            print(f"Running {script_name}...")
            self.run_script(script_path)