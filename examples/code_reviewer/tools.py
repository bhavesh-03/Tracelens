"""Tools used by the Sub-Agents in the Code Reviewer application."""

import ast
import re

def run_ast_linter(code: str) -> str:
    """Parses Python code into an AST to check for basic syntax errors."""
    try:
        ast.parse(code)
        return "PASS: No syntax errors detected by AST parser."
    except SyntaxError as e:
        return f"FAIL: SyntaxError at line {e.lineno}, offset {e.offset}: {e.msg}\n{e.text}"
    except Exception as e:
        return f"FAIL: Unexpected error during parsing: {e}"


def run_security_scanner(code: str) -> str:
    """Uses Regex rules to find obvious vulnerabilities."""
    issues = []
    
    # Check for hardcoded secrets
    if re.search(r'(api_key|password|secret|token)\s*=\s*[\'"][a-zA-Z0-9_\-]+[\'"]', code, re.IGNORECASE):
        issues.append("Hardcoded secret or API key detected.")
        
    # Check for dangerous builtins
    if re.search(r'\b(eval|exec)\(', code):
        issues.append("Dangerous use of `eval()` or `exec()` detected.")
        
    # Check for raw SQL interpolation
    if re.search(r'SELECT.*FROM.*%.*s', code, re.IGNORECASE) or re.search(r'SELECT.*FROM.*f[\'"]', code, re.IGNORECASE):
        issues.append("Possible SQL Injection vulnerability (string formatting in SQL query).")

    if not issues:
        return "PASS: No obvious security vulnerabilities found."
    
    return "FAIL: Security issues found:\n- " + "\n- ".join(issues)
