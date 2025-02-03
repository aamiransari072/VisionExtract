import google.generativeai as genai


class Gemini:
    def __init__(self,api_key='api_key',id='gemini-1.5-flash-latest',temprature=0.2,**kwargs):
        self.api_key = api_key
        self.id = id
        genai.configure(api_key = self.api_key)
        self.model = genai.GenerativeModel(
            self.id,
            generation_config=genai.GenerationConfig(
                temperature=temprature,
                **kwargs
            
            ))
    
    def generate(self,prompt):
        return self.model.generate_content([prompt])




    