import ast
import inspect
from radon.metrics import h_visit
from radon.complexity import cc_visit

class CodeEvaluator:
    @staticmethod
    def complexity_analysis(code):
        """Analyze code complexity"""
        try:
            complexity = cc_visit(code)
            return {
                'max_complexity': max(complexity, key=lambda x: x.complexity).complexity if complexity else 0,
                'average_complexity': sum(c.complexity for c in complexity) / len(complexity) if complexity else 0
            }
        except Exception:
            return {'max_complexity': 0, 'average_complexity': 0}
    
    @staticmethod
    def code_quality_score(code):
        """Basic code quality scoring"""
        try:
            # Parse the code
            tree = ast.parse(code)
            
            # Basic metrics
            function_count = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            comment_lines = len([line for line in code.split('\n') if line.strip().startswith('#')])
            
            return {
                'function_count': function_count,
                'comment_ratio': comment_lines / len(code.split('\n')) if code.split('\n') else 0
            }
        except Exception:
            return {'function_count': 0, 'comment_ratio': 0}