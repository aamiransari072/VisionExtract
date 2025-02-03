import os
import shutil 
import random 
from sklearn.model_selection import train_test_split




source_dir = r"D:\Projects\Assignmenr\artifacts\Data\docs-sm"
target_dir = r"D:\Projects\Assignmenr\data"


train_ratio = 0.8
test_ratio = 0.2

os.makedirs(os.path.join(target_dir,"train"),exist_ok=True)
os.makedirs(os.path.join(target_dir,"test"),exist_ok=True)



for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir,class_name)

    if not os.path.isdir(class_path):
        continue


    os.makedirs(os.path.join(target_dir,"train",class_name),exist_ok=True)
    os.makedirs(os.path.join(target_dir,"test",class_name),exist_ok=True)

    images = [img for img in os.listdir(class_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)


    train_file , test_file = train_test_split(images,test_size=test_ratio)

    for file in train_file:
        shutil.copy2(os.path.join(class_path,file),os.path.join(target_dir,"train",class_name,file))

    for file in train_file:
        shutil.copy2(os.path.join(class_path,file),os.path.join(target_dir,"test",class_name,file))



print("Data Restructuring Complete")


