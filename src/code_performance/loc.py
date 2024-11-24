import os
from radon.metrics import mi_visit

def calculate_loc(file_path):
    """Calculate Lines of Code (LOC) for a given Python file."""
    try:
        with open(file_path, 'r') as file:
            code = file.read()
        lines = len(code.splitlines())
        return {'loc': lines}
    except Exception as e:
        return {'error': str(e)}
