import cv2
import os
import numpy as np
import pdf2image
import matplotlib.pyplot as plt
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import accuracy_score
from tensorflow.keras import callbacks, optimizers

def pdf_to_img(path_of_pdf, filename):
   
    try:
        pdf = pdf2image.convert_from_path(path_of_pdf+'/'+filename)
        i=0
        if 'images' not in os.listdir(path_of_pdf):
            os.makedirs(path_of_pdf +'/images')
        #print(path_of_pdf +'/images')
        for img in pdf:
            img.save(path_of_pdf +'/images/'+filename+str(i)+'.jpg', 'JPEG')
            #print(path_of_pdf +'/images/'+filename+str(i)+'.jpg')
            i = i+1
    except Exception as e:
        pass
        
    images = []
    for img in os.listdir(path_of_pdf +'/images'):
        #print(img)
        if img == '.DS_Store':
            pass
        else:
            pic = image.load_img(path_of_pdf +'/images'+'/'+img, target_size=(300, 300))
            images.append(image.img_to_array(pic))
    

    for item in os.listdir(path_of_pdf +'/images'):
        os.remove(path_of_pdf +'/images'+'/'+item) 
        
    return np.array(images)/255            