import os
from dotenv import load_dotenv  
from groq import Groq
from langchain_groq import ChatGroq

from langchain_groq import ChatGroq

class LLMSelector:
    def __init__(self, providers):
        self.providers = {
            'Groq Llama 3 70B': 'llama-3.1-70b-versatile',
            'Groq Mixtral 8x7B': 'mixtral-8x7b-32768',
            # Add more Groq LLMs
        }
    
    def get_llm(self, api_key, model):
        
        return ChatGroq(
            groq_api_key=api_key,
            model_name=model,
            temperature=0.7
        )