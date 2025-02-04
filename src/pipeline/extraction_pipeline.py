from src.components.data_extraction import DataExtraction
from src.logger import logging  
from src.exception import CustomException  
import sys

class DataExtractionPipeline:
    def __init__(self, img_path):
        self.img_path = img_path
        self.dataExtraction = DataExtraction()

    def run_pipeline(self):
        try:
            logging.info(f"Starting data extraction pipeline for image: {self.img_path}")
            
            data = self.dataExtraction.initiate_data_extraction(img_path=self.img_path)
            
            if not data:
                raise CustomException("No data extracted from the image", sys)
            
            logging.info(f"Data extraction successful for image: {self.img_path}")
            return data
        
        except Exception as e:
            logging.error(f"Error during data extraction for image {self.img_path}: {e}")
            raise CustomException(f"Error during data extraction for image {self.img_path}: {e}", sys) from e
