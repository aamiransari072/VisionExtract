import cv2
import numpy as np
import os

class DataTransformation:
    def __init__(self):
        pass

    def get_images_labels(self,images,label):
        arr = []
        labels = []
        for i in images:
            img = cv2.imread(os.path.join(i))
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
            arr.append(img)
            labels.append(label)
        return [arr,labels]
    

    




