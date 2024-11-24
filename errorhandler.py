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

    def update_file(self, file_path: str, new_content: str) -> bool:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        except Exception as e:
            print(f"Error updating file {file_path}: {str(e)}")
            return False

    def get_llm_fix(self, error_info: Dict[str, str], file_content: str) -> Optional[str]:
        prompt = PromptTemplate(
            input_variables=["error_type", "full_error", "file_content"],
            template="""
            Fix the following Python code that produced this error:

            Error Type: {error_type}
            Full Error:
            {full_error}

            Current code:
            {file_content}

            Provide only the complete fixed code, ready to save to a file.
            """
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)

        for attempt in range(self.max_retries):
            try:
                response = chain.run({
                    "error_type": error_info['error_type'],
                    "full_error": error_info['full_error'],
                    "file_content": file_content
                })
                print (response)
                return response['text']  # Adjust based on actual response structure.
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(self.retry_delay)

        return None
