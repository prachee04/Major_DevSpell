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
    def __init__(self, llm_model: str, max_retries: int = 2, retry_delay: int = 2):
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

    def _filter_nltk_messages(self, error_msg: str) -> str:
        """
        Filter out NLTK download messages from error output.
        """
        # Split the error message into lines
        lines = error_msg.split('\n')
        
        # Filter out NLTK download related messages
        filtered_lines = [
            line for line in lines 
            if not (
                '[nltk_data]' in line or 
                'Downloading package' in line or 
                'Package' in line and 'is already up-to-date!' in line
            )
        ]
        
        # Rejoin the filtered lines
        return '\n'.join(filtered_lines).strip()

    def parse_error(self, error_msg: str) -> Dict[str, Optional[str]]:
        # First filter out NLTK messages
        filtered_error = self._filter_nltk_messages(error_msg)
        
        file_match = re.search(r'File "([^"]+)"', filtered_error)
        line_match = re.search(r'line (\d+)', filtered_error)
        error_type_match = re.search(r'([A-Za-z.]+Error:?.*?)(?:\n|$)', filtered_error)

        return {
            "file_path": file_match.group(1) if file_match else None,
            "line_number": line_match.group(1) if line_match else None,
            "error_type": error_type_match.group(1) if error_type_match else None,
            "full_error": filtered_error
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
        Current code:
        {file_content}
        Provide only the complete fixed code, ready to save to a file.
        Do not add any comments
        """

        for attempt in range(self.max_retries):
            try:
                print(f"Attempt {attempt + 1}: Sending request to the ChatGPT model...")
                # print(error_info['error_type'])
                print(prompt)
                # print(error_info['error_type'])
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an expert ai/ml developer for debugging Python code."},
                        {"role": "user", "content": prompt}
                    ],
                    model="gpt-4o",
                    temperature=0.7,
                    max_tokens=4096,
                    top_p=1,
                )
                fixed_code = response.choices[0].message.content
                fixed_code = self._sanitize_output(fixed_code)
                # print(fixed_code)
                # Assuming the fixed code is returned in the 'choices[0].message.content'
                if response:
                    return fixed_code
                else:
                    print("Error: Invalid response structure.")
                    return None

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(self.retry_delay)

        print("Failed to get a fix after all retry attempts.")
        return None