#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder


# In[35]:


df = pd.read_excel("data.xls")


# In[43]:


df = df.replace(np.nan, 0)


# In[44]:


df


# In[24]:


df = df.iloc[:,2:19]
df


# In[25]:


df.describe()


# In[14]:


A = df.iloc[:,0:3]
A


# In[15]:


A.describe()


# In[16]:


B = df.iloc[:,3:6]
B


# In[17]:


B.describe()


# In[18]:


C = df.iloc[:,6:9]
C


# In[19]:


C.describe()


# In[20]:


D = df.iloc[:,9:12]
D


# In[21]:


D.describe()


# In[39]:


E = df.iloc[:,14:19]
E


# In[40]:


E.describe()

