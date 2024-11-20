from groq import Groq
import os 

class LLMSelector:
    def __init__(self, llm_providers):
        self.llm_providers = llm_providers
        self.clients = {
            'groq_llama3_70b': self._init_groq_llama3(),
            'groq_mixtral_8x7b': self._init_groq_mixtral(),
            'groq_gemma_7b': self._init_groq_gemma()
        }
    
    def _init_groq_llama3(self):
        return Groq(
            api_key=os.getenv('GROQ_API_KEY'),
            model="llama3-70b-8192"
        )
    
    def _init_groq_mixtral(self):
        return Groq(
            api_key=os.getenv('GROQ_API_KEY'),
            model="mixtral-8x7b-32768"
        )
    
    def _init_groq_gemma(self):
        return Groq(
            api_key=os.getenv('GROQ_API_KEY'),
            model="gemma-7b-it"
        )
    
    def get_llm(self, provider):
        return self.clients.get(provider)