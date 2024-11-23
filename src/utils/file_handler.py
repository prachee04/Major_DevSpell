import subprocess
import os

def run_script(script_path, dataset_path):
    """
    Run a Python script with the given dataset as input.

    Args:
        script_path (str): Path to the Python script.
        dataset_path (str): Path to the dataset file.

    Returns:
        dict: Script execution output including stdout and stderr.
    """
    try:
        command = f"python {script_path} {dataset_path}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "returncode": 1,
        }
