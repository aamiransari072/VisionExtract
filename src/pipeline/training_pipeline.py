from src.components.model_trainer import ModelTraining
from keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import to_categorical
from pathlib import Path
import numpy as np
from src.components.data_transformation import DataTransformation

class TrainingPipeline:
    def __init__(self):
        self.modeltrainer = ModelTraining()
        self.transformer = DataTransformation()
        self.earlyStopping = EarlyStopping(monitor = 'loss', patience = 16, mode = 'min', restore_best_weights = True)
        self.train_data_path = Path(r"D:\Projects\Assignmenr\data\train")

    
    def initiate_model_training(self):
        [image1,label1] = self.transformer.get_images_labels(list(self.train_data_path.glob("invoice/*.*")), 0)
        [image2,label2]= self.transformer.get_images_labels(list(self.train_data_path.glob("budget/*.*")), 1)
        [image3,label3] = self.transformer.get_images_labels(list(self.train_data_path.glob("form/*.*")), 2)
        images = np.concatenate((image1,image2,image3),axis=0)
        labels = np.concatenate((label1,label2,label3),axis=0)
        images = np.asarray(images)
        labels = np.asarray(labels)
        labels = to_categorical(labels)
        model = self.modeltrainer.compile_model()
        model = self.modeltrainer.train_model(
            model=model,
            images=images,
            labels=labels,
            epochs=100,
            batch_size=32,
            callbacks=[self.earlyStopping]
        )

        output_path = Path("D:/Projects/Assignmenr/outputs")
        output_path.mkdir(parents=True, exist_ok=True)
        model.save(output_path / "model.h5")
        print("Training Completed")


