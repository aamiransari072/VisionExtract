import os
import cv2
import re
import pytesseract
from src.Agent.google import Gemini
from src.configuration.config import Configuration
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


class DataExtraction:
    def __init__(self):
        self.model = Gemini(
            api_key=GOOGLE_API_KEY,

        )
        self.config = Configuration()

    def extract_text_from_image(self,img_path):
        try:
            image = cv2.imread(img_path)
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            text = pytesseract.image_to_string(gray,lang='eng')
            # print(f"Extracted Data: {text}")
            return text
        except Exception as e:
            print(e)
            return None
    

    def clean_text(self,text):
        try:
            text = text.replace("\n", " ")  
            text = re.sub(r'\s+', ' ', text).strip()  
            # print(f"Cleaned Data: {text}")
            return text
        
        except Exception as e:
            print(e)
            return None
    

    def get_extracted_data(self,text):
        try:
            prompt = self.config.get_prompt()
            response = self.model.generate(prompt=prompt+text)
            return response.text
        except Exception as e:
            print(e)
            return None
        
    

    def initiate_data_extraction(self,img_path):
        try:
            text = self.extract_text_from_image(img_path=img_path)
            cleaned_text = self.clean_text(text=text)
            data = self.get_extracted_data(cleaned_text)
            return data
        except Exception as e:
            print(e)
            return None
    
    



