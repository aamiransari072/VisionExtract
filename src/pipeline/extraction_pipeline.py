from src.components.data_extraction import DataExtraction



class DataExtractionPipeline:
    def __init__(self,img_path):
        self.img_path = img_path
        self.dataExtraction = DataExtraction()

    def run_pipeline(self):
        data = self.dataExtraction.initiate_data_extraction(img_path=self.img_path)
        return data




