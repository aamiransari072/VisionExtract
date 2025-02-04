import os
import sys
import cv2
import re
import pytesseract
from src.Agent.google import Gemini
from src.configuration.config import Configuration
import json
from dotenv import load_dotenv
from src.logger import logging
from src.exception import CustomException
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
            logging.info(f"Starting text extraction from image: {img_path}")
            image = cv2.imread(img_path)
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            text = pytesseract.image_to_string(gray,lang='eng')
            logging.info(f"Text extraction successful from image: {img_path}")
            return text
        except Exception as e:
            logging.error(f"Error in extract_text_from_image: {e}")
            raise CustomException(f"Error extracting text from image {img_path}", sys) from e
    

    def clean_text(self,text):
        try:
            logging.info("Cleaning extracted text.")
            text = text.replace("\n", " ")  
            text = re.sub(r'\s+', ' ', text).strip()  
            # print(f"Cleaned Data: {text}")
            logging.info("Text cleaning successful.")
            return text
        
        except Exception as e:
            logging.error(f"Error in clean_text: {e}")
            raise CustomException("Error cleaning text", sys) from e


    def get_extracted_data(self,text):
        try:
            logging.info("Generating extracted data using the model.")
            prompt = self.config.get_prompt()
            response = self.model.generate(prompt=prompt+text)
            logging.info("Data extraction from model successful.")
            return response.text
        except Exception as e:
            logging.error(f"Error in get_extracted_data: {e}")
            raise CustomException("Error generating extracted data", sys) from e
        
    

    def initiate_data_extraction(self,img_path):
        try:
            logging.info(f"Initiating data extraction process for image: {img_path}")
            text = self.extract_text_from_image(img_path=img_path)
            cleaned_text = self.clean_text(text=text)
            data = self.get_extracted_data(cleaned_text)
            logging.info(f"Data extraction completed successfully for image: {img_path}")
            return data
        except Exception as e:
            logging.error(f"Error in initiate_data_extraction: {e}")
            raise CustomException(f"Error during data extraction for image {img_path}", sys) from e

    
    



