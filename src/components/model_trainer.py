from src.ml.model import ModelArchitecture
from src.components.data_transformation import DataTransformation
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, Precision, Recall, AUC

class ModelTraining:
    def __init__(self):
        self.model = ModelArchitecture().get_model()

    
    def compile_model(self,optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'],
              ):
        """
        This method is responsible for Compiling model 
        """
        
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.model.compile(optimizer=self.optimizer,loss=self.loss,metrics=self.metrics)
        return self.model
    
    def train_model(self,model,images=None,labels=None,epochs=10,batch_size=32,callbacks=None):
        self.images = images
        self.labels = labels
        self.epochs = epochs
        self.batch_size=batch_size
        self.callbacks = callbacks

        model.fit(
            self.images,
            self.labels,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks= self.callbacks
        )

        return self.model
    

    
    


