import os
import re
import time
import traceback
from typing import Optional, Dict
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

class LLMErrorHandler:
    def __init__(self, llm_model: str, max_retries: int = 3, retry_delay: int = 2):
        self.client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=os.environ["GITHUB_TOKEN"],
        )
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="gemma2-9b-it",
            temperature=0.7,
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def parse_error(self, error_msg: str) -> Dict[str, Optional[str]]:
        file_match = re.search(r'File "([^"]+)"', error_msg)
        line_match = re.search(r'line (\d+)', error_msg)
        error_type_match = re.search(r'([A-Za-z.]+Error:?.*?)(?:\n|$)', error_msg)

        return {
            "file_path": file_match.group(1) if file_match else None,
            "line_number": line_match.group(1) if line_match else None,
            "error_type": error_type_match.group(1) if error_type_match else None,
            "full_error": error_msg
        }

    def read_file_content(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return ""
    def _sanitize_output(self, text):
        """
        Extract and return only valid Python code from the text.
        Removes any non-Python content, including markdown, extra formatting,
        and irrelevant text.
        """
        # Regular expression to match Python code blocks
        python_code_blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
        
        if python_code_blocks:
            # Concatenate all detected Python code blocks
            sanitized_code = "\n".join(python_code_blocks)
        else:
            # If no explicit Python code blocks, assume the entire text might be code
            # but filter out anything obviously not Python (like markdown headers)
            sanitized_code = re.sub(r"[^a-zA-Z0-9_#:\n\(\)\[\]\{\}.,=+\-*\/<>%&|! ]", "", text)
        
        return sanitized_code.strip()
    def update_file(self, file_path: str, new_content: str) -> bool:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        except Exception as e:
            print(f"Error updating file {file_path}: {str(e)}")
            return False

    def get_llm_fix(self, error_info: Dict[str, str], file_content: str) -> Optional[str]:
        """
        Sends a request to the AI model to fix the code error and returns the corrected code.
        """
        prompt = f"""
        Fix the following Python code that produced this error:
        Error Type: {error_info['error_type']}
        Full Error:
        {error_info['full_error']}
        Current code:
        {file_content}
        Provide only the complete fixed code, ready to save to a file.
        """

        for attempt in range(self.max_retries):
            try:
                print(f"Attempt {attempt + 1}: Sending request to the ChatGPT model...")
                
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an assistant for debugging Python code."},
                        {"role": "user", "content": prompt}
                    ],
                    model="gpt-4o",
                    temperature=0.7,
                    max_tokens=4096,
                    top_p=1,
                )
                fixed_code = response.choices[0].message.content
                fixed_code =self._sanitize_output(fixed_code)
                # Assuming the fixed code is returned in the 'choices[0].message.content'
                if response :
                    return fixed_code
                else:
                    print("Error: Invalid response structure.")
                    return None

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(self.retry_delay)

        print("Failed to get a fix after all retry attempts.")
        return None


