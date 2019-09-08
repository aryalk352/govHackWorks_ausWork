#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import numpy as np


# In[5]:


dataset_train = pd.read_csv("prediction_file_train.csv")


# In[6]:


dataset_train


# In[7]:


training_set = dataset_train.iloc[:,1:2].values


# In[8]:


training_set


# In[9]:


from sklearn.preprocessing import MinMaxScaler


# In[11]:


sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)


# In[12]:


training_set_scaled


# In[13]:


x_train = []
y_train = []
for i in range (60, 1258):
    x_train.append(training_set_scaled[i-60:i])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)


# In[14]:


x_train


# In[15]:


from keras.models import Sequential


# In[16]:


from keras.layers import Dense


# In[17]:


from keras.layers import LSTM


# In[18]:


from keras.layers import Dropout


# In[19]:


regressor = Sequential()


# In[24]:


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))


# In[25]:


regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


# In[26]:


regressor.add(Dense(units = 1))


# In[28]:


regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[29]:


regressor.fit(x_train, y_train, epochs =100, batch_size = 32)


# In[31]:


dataset_test = pd.read_csv("prediction_file_test.csv")
real_predection = dataset_test.iloc[:, 1:2].values


# In[32]:


dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)


# In[33]:


inputs = dataset_total[len(dataset_total - len (dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
                           dataset_total[len(dataset_total - len (dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
x_test = []
for i in range (60, 80):
	x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[i], 1))
predicted_prices = regressor.predict(x_test)
predicted_prices = sc.inverse_transform(predicted_prices)


# In[ ]:




