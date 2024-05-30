import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model("VGG19.h5")

image=cv2.imread('D:\project\datasets\BrownSpot\IMG_2992.jpg')
# image=cv2.imread('/Users/shekarri/Desktop/project/datasets/ModerateDemented/moderateDem90.jpg')
# image=cv2.imread('/Users/shekarri/Desktop/project/datasets/NonDemented/nonDem10.jpg')
# image=cv2.imread('/Users/shekarri/Desktop/project/datasets/VeryMildDemented/verymildDem0.jpg')



img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result=model.predict(input_img)


print(result)
