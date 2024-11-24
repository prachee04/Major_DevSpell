from radon.metrics import mi_visit

def calculate_mi(file_path):
    """Calculate Maintainability Index (MI) for a given Python file."""
    try:
        with open(file_path, 'r') as file:
            code = file.read()
        maintainability = mi_visit(code, multi=True)
        return {'mi': maintainability}
    except Exception as e:
        return {'error': str(e)}
