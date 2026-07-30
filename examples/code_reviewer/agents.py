"""Multi-Agent Code Reviewer (Subject Application for TraceLens)."""


import litellm

from tracelens.capture import TraceLensCapture

from .tools import run_ast_linter, run_security_scanner

# Default lightweight model for testing
DEFAULT_MODEL = "gemini/gemini-2.5-flash"


def llm_call(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Helper to make a simple litellm completion call."""
    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
        )
        return str(response.choices[0].message.content)
    except litellm.exceptions.APIConnectionError:
        # Fallback for local testing without an API key
        return f"[MOCK LLM RESPONSE]: Processed prompt for {model}. Output looks good."
    except Exception as e:
        return f"[MOCK LLM RESPONSE]: Fallback due to error: {e}"


class LinterAgent:
    def __init__(self, tracer: TraceLensCapture):
        self.tracer = tracer

    def review(self, code: str) -> str:
        """Runs the AST linter and generates a syntax report."""
        with self.tracer.step(
            agent_name="LinterAgent", 
            step_type="agent", 
            input_text=code, 
            model=DEFAULT_MODEL
        ) as io:
            
            # Sub-step: Tool call
            with self.tracer.step(
                agent_name="AST Linter Tool", 
                step_type="tool", 
                input_text=code,
                tool_name="run_ast_linter"
            ) as tool_io:
                tool_result = run_ast_linter(code)
                tool_io.output_text = tool_result
            
            # LLM Analysis
            prompt = f"Analyze this syntax report and summarize it for the developer:\n\n{tool_result}"
            report = llm_call("You are a strict syntax reviewer.", prompt)
            
            io.output_text = report
            return report


class SecurityAgent:
    def __init__(self, tracer: TraceLensCapture):
        self.tracer = tracer

    def review(self, code: str) -> str:
        """Runs the security scanner and generates a vulnerability report."""
        with self.tracer.step(
            agent_name="SecurityAgent", 
            step_type="agent", 
            input_text=code, 
            model=DEFAULT_MODEL
        ) as io:
            
            # Sub-step: Tool call
            with self.tracer.step(
                agent_name="Regex Scanner Tool", 
                step_type="tool", 
                input_text=code,
                tool_name="run_security_scanner"
            ) as tool_io:
                tool_result = run_security_scanner(code)
                tool_io.output_text = tool_result
            
            # LLM Analysis
            prompt = f"Code:\n```python\n{code}\n```\nScanner Result:\n{tool_result}\n\nWrite a brief security review."
            report = llm_call("You are an application security auditor.", prompt)
            
            io.output_text = report
            return report


class PerformanceAgent:
    def __init__(self, tracer: TraceLensCapture):
        self.tracer = tracer

    def review(self, code: str) -> str:
        """Analyzes code for Big-O performance issues using pure LLM reasoning."""
        with self.tracer.step(
            agent_name="PerformanceAgent", 
            step_type="agent", 
            input_text=code, 
            model=DEFAULT_MODEL
        ) as io:
            
            prompt = f"Analyze the time and space complexity of this code:\n\n```python\n{code}\n```"
            report = llm_call("You are a performance optimization expert.", prompt)
            
            io.output_text = report
            return report


class SynthesizerAgent:
    def __init__(self, tracer: TraceLensCapture):
        self.tracer = tracer

    def merge_reports(self, code: str, reports: list[str]) -> str:
        """Merges sub-reports into a final Code Review document."""
        input_data = f"Code:\n{code}\n\nReports:\n" + "\n---\n".join(reports)
        
        with self.tracer.step(
            agent_name="SynthesizerAgent", 
            step_type="synthesizer", 
            input_text=input_data, 
            model=DEFAULT_MODEL
        ) as io:
            
            prompt = f"Synthesize these code review reports into a cohesive final markdown summary.\n\n{input_data}"
            final_report = llm_call("You are the Lead Code Reviewer.", prompt)
            
            io.output_text = final_report
            return final_report


class CoordinatorAgent:
    def __init__(self, tracer: TraceLensCapture):
        self.tracer = tracer
        self.linter = LinterAgent(tracer)
        self.security = SecurityAgent(tracer)
        self.performance = PerformanceAgent(tracer)
        self.synthesizer = SynthesizerAgent(tracer)

    def run_review(self, code: str) -> str:
        """Orchestrates the multi-agent code review process."""
        with self.tracer.step(
            agent_name="CoordinatorAgent", 
            step_type="router", 
            input_text=code
        ) as io:
            
            # Run sub-agents sequentially (could be async/parallel)
            r1 = self.linter.review(code)
            r2 = self.security.review(code)
            r3 = self.performance.review(code)
            
            # Synthesize
            final_report = self.synthesizer.merge_reports(code, [r1, r2, r3])
            
            io.output_text = final_report
            return final_report
