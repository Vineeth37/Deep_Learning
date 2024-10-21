#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[2]:


df = sns.load_dataset('iris')
df.head()


# In[3]:


df['species'].unique()


# In[4]:


df['species'] = df['species'].map({'setosa' : 0,'versicolor' : 1 , 'virginica' : 2}).astype(int)


# In[5]:


df.sample(5)


# In[6]:


X = df.iloc[ : , :-1]
y = df.iloc[ : ,-1]


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# In[9]:


len(X_train) , len(y_train)


# In[10]:


len(X_test), len(y_test)


# In[11]:


get_ipython().system('pip install tensorflow')


# In[17]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  # Dense -> Hidden layer
from tensorflow.keras.activations import sigmoid,relu


# In[18]:


X.shape[1]


# In[19]:


import warnings
warnings.filterwarnings('ignore')


# In[20]:


model = Sequential() # Sequential is used to build architectures and it is a class_> model is an object for that

model.add(Dense(units = 64,kernel_initializer='he_uniform',activation ='relu', input_dim = X.shape[1])) #HL1

model.add(Dense(units = 32,kernel_initializer='he_uniform',activation ='relu')) #HL2

model.add(Dense(units = 16,kernel_initializer='he_uniform',activation ='relu')) #HL3

model.add(Dense(units = 4,kernel_initializer='he_uniform',activation ='relu')) #HL4

model.add(Dense(units = 3,kernel_initializer='glorot_uniform',activation ='softmax')) #HL1



# In[21]:


model.summary()


# In[23]:


model.compile(optimizer ='adam',metrics = ['accuracy'] , loss = 'categorical_crossentropy')


# In[25]:


y_train[:5]


# In[26]:


y_train_p = tf.keras.utils.to_categorical(y_train,num_classes=3)
y_train_p[:5]


# In[29]:


# Now i will give above data to the architecture

model.fit(X_train,y_train_p,epochs=50 , batch_size = 15 , validation_split = 0.2)


# In[30]:


model.history.history.keys()


# In[31]:


model.history.history['accuracy']


# In[32]:


model.history.history['loss']


# In[33]:


model.history.history['val_accuracy']


# In[34]:


model.history.history['val_loss']


# In[36]:


plt.figure(figsize = (10,3))
plt.subplot(1,2,1)
plt.title('Train Performance')
plt.plot(np.arange(1,51),model.history.history['accuracy'],color = 'g' ,label = 'train_acc')
plt.plot(np.arange(1,51),model.history.history['loss'],color = 'r' ,label = 'train_loss')

plt.subplot(1,2,2)
plt.title('Validation Performance')
plt.plot(np.arange(1,51),model.history.history['val_accuracy'],color = 'g' ,label = 'val_acc')
plt.plot(np.arange(1,51),model.history.history['val_loss'],color = 'r' ,label = 'val_loss')

plt.legend(loc = 0)
plt.show()


# In[37]:


X_test.head(1)


# In[38]:


y_test.head(1)


# In[39]:


d = []
for i in X_test.columns:
    d.append(X_test[i][73])
print(d)


# In[42]:


d = np.array(d).reshape(1,-1)
d.shape


# In[43]:


# Giving the data to the trained NN

model.predict(d)


# In[47]:


labels = ['setosa','versicolor','virginica']


# In[48]:


labels[np.argmax(model.predict(d))]


# In[44]:


y_test_pred = model.predict(X_test)


# In[49]:


y_test_pred


# In[53]:


c = []
for i in y_test_pred:
    if np.argmax(i) == 0:
        c.append(0)
    elif np.argmax(i) == 1:
        c.append(1)
    else:
        c.append(2)


# In[54]:


c


# In[55]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[56]:


confusion_matrix(y_test,c)


# In[57]:


accuracy_score(y_test,c)


# In[ ]:




