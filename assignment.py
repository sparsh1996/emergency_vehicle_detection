#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import models, layers, optimizers, regularizers, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Data gathering and preprocessing

# #### The dataset used here consists of the images of the vehicles, and the task here is to classify them either as 'Emergency vehicle', or not. The number of images used here will be reduced to 800.

# In[2]:


path = '/Users/skylark/Desktop/Misc/cv/train_SOaYf6m/'
data = pd.read_csv('/Users/skylark/Desktop/Misc/cv/train_SOaYf6m/train.csv')


# In[9]:


new_data = data.sample(800)


# In[12]:


new_data.reset_index(drop=True, inplace=True)


# In[13]:


new_data


# In[15]:


images = []
for img in new_data.image_names:
    path_img = path+'images/'+img
    images.append(image.img_to_array(image.load_img(path=path_img, target_size=(224,224))))


# In[21]:


images = np.array(images)
images.shape


# In[20]:


image.load_img(path=path_img, target_size=(224,224))


# In[22]:


target = new_data.emergency_or_not


# In[25]:


train_x, test_x, train_y, test_y = train_test_split(images, target, random_state = 42, test_size=0.3)


# ### Building ResNet model

# #### The resnet model will be trained using only 560

# In[41]:


len(train_x)


# In[23]:


resnet = ResNet50(include_top=False, input_shape=(224,224,3))


# In[27]:


flag = False

for layer in resnet.layers:
    if layer.name in ['conv5_block3_1_conv','conv5_block3_1_bn','conv5_block3_1_relu','conv5_block3_2_conv','conv5_block3_2_bn',
 'conv5_block3_2_relu','conv5_block3_3_conv','conv5_block3_3_bn','conv5_block3_add','conv5_block3_out']:
        flag = True
            
    if flag == True:
        layer.trainable = True
    else:
        layer.trainable = False


# In[29]:


x = layers.Flatten()(resnet.output)
final = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs = resnet.input, outputs = final)
model.summary()


# In[30]:


model.compile(optimizer=optimizers.RMSprop(2e-5), loss='binary_crossentropy', metrics=['acc'])
history = model.fit(train_x, train_y, batch_size=32, epochs=10, validation_split=0.2)


# In[35]:


history.history.keys()


# In[36]:


loss = history.history['loss']
val_loss = history.history['val_loss']

acc = history.history['acc']
val_acc = history.history['val_acc']


# In[39]:


plt.plot(range(1,11), loss, label='Training_loss')
plt.plot(range(1,11), val_loss, label='Validation_loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# In[40]:


plt.plot(range(1,11), acc, label='Training_acc')
plt.plot(range(1,11), val_acc, label='Validation_acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


# In[31]:


model.evaluate(test_x, test_y)


# #### The Resnet model trained using only 560 images, gave an accuracy of 89.999..% on the validation set.
