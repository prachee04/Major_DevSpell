import os
from dotenv import load_dotenv  
from groq import Groq

class LLMSelector:
    def __init__(self, llm_providers):
        # Load environment variables
        load_dotenv()
        
        self.llm_providers = llm_providers
        self.clients = {
            'groq_llama3_70b': self._init_groq_llama3(),
            'groq_mixtral_8x7b': self._init_groq_mixtral(),
            'groq_gemma_7b': self._init_groq_gemma()
        }
    
    def _init_groq_llama3(self):
        return Groq(api_key=os.getenv('GROQ_API_KEY'))
    
    def _init_groq_mixtral(self):
        return Groq(api_key=os.getenv('GROQ_API_KEY'))
    
    def _init_groq_gemma(self):
        return Groq(api_key=os.getenv('GROQ_API_KEY'))
    
    def get_llm(self, provider):
        client = self.clients.get(provider)
        
        model_mapping = {
            'groq_llama3_70b': "llama3-70b-8192",
            'groq_mixtral_8x7b': "mixtral-8x7b-32768",
            'groq_gemma_7b': "gemma-7b-it"
        }
        
        return client, model_mapping.get(provider)