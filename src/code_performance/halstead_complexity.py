from radon.metrics import h_visit

def calculate_hc(file_path):
    """Calculate Halstead Complexity (HC) for a given Python file."""
    try:
        with open(file_path, 'r') as file:
            code = file.read()
        halstead_metrics = h_visit(code)
        return halstead_metrics  # Includes multiple Halstead metrics
    except Exception as e:
        return {'error': str(e)}
