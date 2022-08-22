from tensorflow.keras.models import load_model 
import cv2 
import numpy as np 
def Function(img_path, model_path):
  img = cv2.imread(img_path)
  img = cv2.resize(img,(100,100))
  img = img/255.0
  model = load_model(model_path)
  img = np.array([img])
  if np.argmax(model.predict(img)) == 3:
    return {"Classified_Class":"Nothing Detected"}
  else:
    return {"Classified_Class":f"{np.argmax(model.predict(img))}"}


path_to_img = '/content/17db9286303c9030e93296b704cb8fe8.jpg'
path_to_model = '/content/drive/MyDrive/furniture/Resnet50V2_82p.h5'
print(Function(path_to_img,path_to_model))