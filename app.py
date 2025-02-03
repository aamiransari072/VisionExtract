# import cv2
# import pytesseract
# from pathlib import Path

# # Load the image

# image = cv2.imread(image_path)

# # Convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply thresholding for better OCR accuracy
# gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# # Perform OCR using Tesseract
# text = pytesseract.image_to_string(gray,lang='eng')

# print("Extracted Text:\n", text)



# from src.pipeline.extraction_pipeline import DataExtractionPipeline

# img_path = r"D:\Projects\Assignmenr\artifacts\Data\docs-sm\invoice\00920638.jpg"

# pipeline = DataExtractionPipeline(img_path=img_path)
# response = pipeline.run_pipeline()

# print(response)


# from src.ml.model import ModelArchitecture

# model = ModelArchitecture()

# model = model.get_model()
# print(model.summary())






# pipeline  = TraningPipeline()

# pipeline.initiate_model_training()


from src.pipeline.training_pipeline import TraningPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.extraction_pipeline import DataExtractionPipeline
from fastapi import FastAPI
import uvicorn
import numpy as np
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import tempfile
import os

def save_image_and_get_path(img_data):
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, 'uploaded_image.jpg')
    cv2.imwrite(temp_path, img)
    return temp_path

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    try:
        train_pipeline = TraningPipeline()

        train_pipeline.initiate_model_training()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    

@app.post("/predict")
async def predict_route(image: UploadFile = File(...)):
    try:
        img_data = await image.read()
        obj = PredictionPipeline()
        result = obj.run_pipeline(img_data)
        predicted_class = int(np.argmax(result))
        return JSONResponse(content={"predicted_class": predicted_class})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/classify_and_extract")
async def classify_and_extract(image: UploadFile = File(...)):
    try:
        img_data = await image.read()
        obj = PredictionPipeline()
        result = obj.run_pipeline(img=img_data)
        pred_class = int(np.argmax(result))

        if pred_class == 0:
            image_path = save_image_and_get_path(img_data) 
            extraction_obj = DataExtractionPipeline(image_path)
            response = extraction_obj.run_pipeline()
            return JSONResponse(content={"predicted_class": pred_class, "extracted_data": response})

        return JSONResponse(content={"predicted_class": pred_class, "message": "No extraction needed"})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__=="__main__":
    uvicorn.run(app)
    







