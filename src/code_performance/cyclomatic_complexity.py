from radon.complexity import cc_visit

def calculate_cc(file_path):
    """Calculate Cyclomatic Complexity (CC) for a given Python file."""
    try:
        with open(file_path, 'r') as file:
            code = file.read()
        complexity_results = cc_visit(code)
        total_cc = sum(block.complexity for block in complexity_results)
        return {'cc': total_cc}
    except Exception as e:
        return {'error': str(e)}
