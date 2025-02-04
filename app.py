from src.logger import logging
from src.exception import CustomException
from src.pipeline.training_pipeline import TrainingPipeline  # Fixed typo
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.extraction_pipeline import DataExtractionPipeline
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.responses import RedirectResponse
import numpy as np
from PIL import Image
import io
import tempfile
import os
import sys
import json
import uuid
import uvicorn

app = FastAPI()

def save_image_and_get_path(img_data):
    """Save binary image data to a temporary file and return its path."""
    try:
        image = Image.open(io.BytesIO(img_data))  
        temp_dir = tempfile.gettempdir()

        # Generate a unique filename to avoid overwriting
        unique_filename = f"uploaded_{uuid.uuid4().hex}.jpg"
        temp_path = os.path.join(temp_dir, unique_filename)

        image.save(temp_path) 
        logging.info(f"Image saved successfully at {temp_path}")
        return temp_path  

    except Exception as e:
        logging.error(f"Error in save_image_and_get_path: {e}")
        raise CustomException(e, sys) from e
    


@app.get("/", tags=["authentication"])
async def index():
    logging.info("Redirected to /docs")
    return RedirectResponse(url="/docs")



@app.get("/train")
async def training():
    """Endpoint to start model training."""
    try:
        logging.info("Model training initiated.")
        train_pipeline = TrainingPipeline()
        train_pipeline.initiate_model_training()
        logging.info("Training completed successfully.")
        return Response("Training successful!")

    except Exception as e:
        logging.error(f"Error occurred during training: {e}")
        raise CustomException(e, sys) from e

@app.post("/predict")
async def predict_route(image: UploadFile = File(...)):
    """Predict the class of an uploaded image."""
    try:
        img_data = await image.read()
        image_path = save_image_and_get_path(img_data)

        logging.info(f"Prediction started for image: {image.filename}")
        obj = PredictionPipeline()
        result = obj.run_pipeline(image_path)
        predicted_class = int(np.argmax(result))

        logging.info(f"Prediction completed: Class {predicted_class}")
        return JSONResponse(content={"predicted_class": predicted_class})

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise CustomException(e, sys) from e
    


@app.post("/classify_and_extract")
async def classify_and_extract(image: UploadFile = File(...)):
    """Classify the image and extract data if needed."""
    try:
        img_data = await image.read()
        image_path = save_image_and_get_path(img_data)

        logging.info(f"Classification started for image: {image.filename}")
        obj = PredictionPipeline()
        result = obj.run_pipeline(image_path)
        pred_class = int(np.argmax(result))
        logging.info(f"Classification completed: Class {pred_class}")

        if pred_class == 0:
            logging.info(f"Data extraction initiated for image: {image.filename}")
            extraction_obj = DataExtractionPipeline(image_path)
            response = extraction_obj.run_pipeline()
            cleaned_response = response.strip('```json\n').strip('\n```')
            logging.info("Data extraction completed successfully.")

            try:
                data  = json.loads(cleaned_response)
                return JSONResponse(content={"predicted_class": pred_class, "extracted_data": data})
            except json.JSONDecodeError:
                logging.error("Error decoding the extracted data into JSON.")
                return JSONResponse(content={"predicted_class": pred_class, "extracted_data":cleaned_response}, status_code=400)

        return JSONResponse(content={"predicted_class": pred_class, "message": "No extraction needed"})

    except Exception as e:
        logging.error(f"Error in classification and extraction: {e}")
        raise CustomException(f"An error occurred during classification and extraction: {e}", sys) from e
    


if __name__ == "__main__":
    logging.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
