#!/usr/bin/env python
# coding: utf-8

# In[50]:


import streamlit as st
from pickle import load
import pandas as pd 
from sklearn.svm import SVC
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[51]:


# giving a title
   
st.title('Bankruptcy Prevention System')



# In[52]:


model= pickle.load(open(r"C:\Users\admin\Desktop\SVC_Model.sav" ,'rb'))


# In[53]:


def Bankruptcy_Prevention(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'Buisness show Bankruptcy'
    else:
      return 'Buisness show Non Bankruptcy'
  


# In[54]:


def main():
    
    
    # getting the input data from the user
    
    industrial_risk = st.sidebar.selectbox('industrial_risk',('0','0.5','1'))
    management_risk = st.sidebar.selectbox('management_risk',('0','0.5','1'))
    financial_flexibility = st.sidebar.selectbox('financial flexibility',('0','0.5','1'))
    credibility = st.sidebar.selectbox('credibility',('0','0.5','1'))
    competitiveness = st.sidebar.selectbox('competitiveness',('0','0.5','1'))
    operating_risk = st.sidebar.selectbox('operating risk',('0','0.5','1'))
   
    
    
    # code for Prediction
    Conclusion = ''
    
    # creating a button for Prediction
    
    if st.button('Analyze'):
        
         Conclusion = Bankruptcy_Prevention ([industrial_risk, management_risk, financial_flexibility, credibility, competitiveness ,operating_risk])
        
        
    st.success(Conclusion)
    
    
    
    
    
if __name__ == '__main__':
    main()
     


# In[ ]:




