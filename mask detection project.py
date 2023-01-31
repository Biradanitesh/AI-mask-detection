#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import matplotlib.pyplot as plt
from imutils import paths

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


dataset=r'C:\Users\Admin\Downloads\dataset-20210925T115750Z-001\dataset/'
imagePaths=list(paths.list_images(dataset))


# In[3]:


imagePaths


# In[4]:


data=[]
labels=[]

for i in imagePaths:
    label=i.split(os.path.sep)[-2]
    labels.append(label)
    image=load_img(i,target_size=(224,224))
    image=img_to_array(image)
    image=preprocess_input(image)
    data.append(image)


# In[5]:


labels


# In[ ]:





# In[6]:


image


# In[ ]:





# In[7]:


data


# In[8]:


data


# In[9]:


data=np.array(data,dtype='float32')
labels=np.array(labels)


# In[10]:


data.shape


# In[ ]:





# In[11]:


labels


# In[12]:


lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)


# In[13]:


train_x,test_x,train_y,test_y=train_test_split(data,labels,test_size=0.20,random_state=10,stratify=labels)


# In[14]:


train_x.shape


# In[15]:


train_y.shape


# In[16]:


test_x.shape


# In[17]:


test_y.shape


# In[18]:


aug=ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode='nearest')


# In[19]:


baseModel= MobileNetV2(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))


# In[20]:


baseModel.summary()


# In[21]:


headModel=baseModel.output
 
headModel=Flatten(name='Flatten')(headModel)
headModel=Dense(128,activation='relu')(headModel)
headModel=Dropout(0.5)(headModel)
headModel=Dense(2,activation='softmax')(headModel)

model=Model(inputs=baseModel.input,outputs=headModel)


# In[22]:


for layer in baseModel.layers:
    layer.trainable=False


# In[23]:


model.summary()


# In[24]:


learning_rate=0.001
Epochs=20
BS=12

opt=Adam(lr=learning_rate,decay=learning_rate/Epochs)
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
H=model.fit(
    aug.flow(train_x,train_y,batch_size=BS),
    steps_per_epoch=len(train_x)//BS,
    validation_data=(test_x,test_y),
validation_steps=len(test_x)//BS,
epochs=Epochs
)
model.save(r'C:\Users\Admin\Downloads\dataset-20210925T115750Z-001\dataset\mobilenet_v2.model')


# In[26]:


predict=model.predict(test_x,batch_size=BS)
predict=np.argmax(predict,axis=1)
print(classification_report(test_y,argmax(axis=1),predict,target_names=lb.classes_))


# In[ ]:




