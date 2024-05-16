#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
import joblib

import pickle
#import os

import pandas as pd    # data preprocessing
import numpy as np      # linear algebra
from sklearn.ensemble import ExtraTreesRegressor
# Load the trained model

# path = os.getcwd()
# print("The current path is:",path)
# os.chdir("/Users/naeem/Downloads/Crude_Data")
# path1 = os.getcwd()
# print("The changed path is:", path1)

X_test  = pd.read_csv('CrudeAllocation_IO_XtestData.csv', index_col=False)
X_train = pd.read_csv('CrudeAllocation_IO_XtrainData.csv', index_col=False)
y_test  = pd.read_csv('CrudeAllocation_IO_ytestData.csv', index_col=False)
y_train = pd.read_csv('CrudeAllocation_IO_ytrainData.csv', index_col=False)

y_train1 = y_train['Total_BottomFlow_MBD'].values
y_test1 = y_test['Total_BottomFlow_MBD'].values

model_pkl_file = "Crude_Allocation_IO.pkl"  
# load model from pickle file
with open(model_pkl_file, 'rb') as file:  
    model = pickle.load(file)


model.fit(X_train,y_train1)

    

st.title('Oil Allocation UAT')
st.markdown(' #### Draft dashboard for the crude allocation project ')

st.header("Inputs for Total Bottom Flow (MBD)")

ABQ2_Flow_MBD = st.number_input('### ABQ 2 Flow (MBD) - min=0; max= 89;', min_value=0.0, max_value=89.0, step=0.1, value=44.0)
ABQ3_Flow_MBD = st.number_input('### ABQ 3 Flow (MBD) - min=0, max= 165', min_value=0.0, max_value=165.0, step=0.1, value=64.4)
ABQ5_Flow_MBD = st.number_input('### ABQ 5 Flow (MBD) - min=0, max= 55', min_value=0.0, max_value=55.0, step=0.1, value= 33.0)
ABQ6_Flow_MBD = st.number_input('### ABQ 6 Flow (MBD) - min=0, max= 185', min_value=0.0, max_value=185.0, step=0.1, value= 127.0)

SYB2_Flowrate_GPM = st.number_input('### SYB 2 Flow (GPM) - min=0, max= 21758', min_value=0.0, max_value=21758.0, step=0.1, value=16750.0)
SYB4_Flowrate_GPM = st.number_input('### SYB 4 Flow (GPM) - min=0, max= 19510', min_value=0.0, max_value=19510.0, step=0.1, value=15179.0)
Avg_H2S_ppm = st.number_input('### Average H2S (ppm) - min=1, max= 65', min_value=0.4, max_value=65.0, step=0.1, value=12.0)
Avg_BottomTemp_F = st.number_input('### Average Bottom Temp (F) - min=120, max= 190', min_value=120.0, max_value=190.0, step=0.1, value=154.0)


if st.button('Predict'):
    # Prepare input data
    input_data = np.array([[ABQ2_Flow_MBD, ABQ3_Flow_MBD, ABQ5_Flow_MBD, ABQ6_Flow_MBD, SYB2_Flowrate_GPM, 
                            SYB4_Flowrate_GPM, Avg_H2S_ppm, Avg_BottomTemp_F]])
    
    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Display prediction
    st.write('## Predicted Bottom Flow (MBD):', prediction[0])


# In[ ]:





# In[ ]:




