import os
import keras
import pickle
import numpy as np
import cv2
import tensorflow as tf

from src.components.data_transformation import DataTransformation


class PredictionPipeline:
    def __init__(self):
        self.model_name = "model.h5"
        self.model_path = os.path.join("output",self.model_name)
        self.transformer = DataTransformation()

    def predict(self,img):
        try:
            model = tf.keras.models.load_model(self.model_path)
            img = cv2.imread(os.path.join(img))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img ,(120,120))
            img1 = img[0:30,0:120]/255
            img2 = img[30:90, 0:60]/255
            img3 = img[30:90, 60:120]/255
            img4 = img[90:120, 0:120]/255

            img = np.array(
                    [cv2.resize(img1,(48,48)),
                    cv2.resize(img2,(48,48)),
                    cv2.resize(img3,(48,48)),
                    cv2.resize(img4,(48,48))]
                )

            img_mean = np.mean(img)
            img = img - img_mean
            img = img /np.std(img)
            image = np.asarray(img)
            image = np.expand_dims(image, axis=-1)
            image = np.expand_dims(image, axis=0)
            pred = model.predict(image)
            return pred
        
        except Exception as e:
            print(e)
            return None

    
    def run_pipeline(self,img):
        try:
            pred = self.predict(img=img)
            return pred
        except Exception as e:
            print(e)
            return None
