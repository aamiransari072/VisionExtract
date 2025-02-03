import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling1D, Dense, TimeDistributed
from tensorflow.keras.activations import relu, sigmoid, softmax



class ModelArchitecture:
    def __init__(self):
        pass
    
    def get_model(self):
        model = Sequential()
        model.add(TimeDistributed(Conv2D(512, 3,activation=relu), input_shape=(4, 48, 48, 1)))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(TimeDistributed(Dropout(0.2)))
        model.add(TimeDistributed(Conv2D(256, 3, activation=relu)))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(TimeDistributed(Dropout(0.2)))
        model.add(TimeDistributed(Flatten()))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(1024, activation=sigmoid))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation=softmax)) 
        return model
    

