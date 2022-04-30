#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("funds-1.csv")


# In[3]:


df.shape


# In[5]:


df.head()


# In[6]:


test = df[df["Released"].isna()]
test


# In[7]:


train  = df.dropna()
train


# In[8]:


X_train = train.drop("Released", axis = 1)
y_train = train.Released

X_test = test.drop("Released", axis = 1)
y_test = test.Released


# In[9]:


print(y_test)


# In[8]:


X_test.shape, y_test.shape


# In[9]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

y_pred


# In[10]:


from sklearn.model_selection import train_test_split

X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

y_val_lin_reg_pred = lin_reg.predict(X_val_test)

y_val_lin_reg_pred


# In[11]:


from sklearn.ensemble import RandomForestRegressor


# In[12]:


RFR = RandomForestRegressor()

RFR.fit(X_val_train, y_val_train)

RFR_pred = RFR.predict(X_val_test)

RFR_pred


# In[13]:


from sklearn.linear_model import BayesianRidge


# In[14]:


BR = BayesianRidge()

BR.fit(X_val_train, y_val_train)

BR_pred = BR.predict(X_val_test)

BR_pred


# In[ ]:


dfff = {"Actual Value" : y_val_test, "Lin_reg Value" : y_val_lin_reg_pred, "RFR" : RFR_pred, "BR" : BR_pred}

Difference = pd.DataFrame(dfff)

Difference


# In[17]:


#Mean square Error
from sklearn.metrics import mean_squared_error, r2_score


# In[19]:


mse=mean_squared_error(y_val_test,y_val_lin_reg_pred)
mse


# In[21]:


r2 = r2_score(y_val_test,y_val_lin_reg_pred)
r2


# In[22]:


mse_rfr = mean_squared_error(y_val_test,RFR_pred)
mse_rfr


# In[23]:


r2_rfr = r2_score(y_val_test,RFR_pred)
r2_rfr


# In[24]:


mse_br = mean_squared_error(y_val_test, BR_pred)
mse_br


# In[25]:


r2_br = r2_score(y_val_test,BR_pred)
r2_br


# In[ ]:




