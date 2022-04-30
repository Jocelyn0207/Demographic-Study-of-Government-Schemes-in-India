#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.metrics import silhouette_score


# In[4]:


data = pd.read_csv("data funds.csv")


# In[5]:


data.head()


# In[6]:


data.shape


# In[7]:


data.isnull().sum()


# In[14]:


data["A"][0]


# In[8]:


data.replace(np.nan, 0)


# In[9]:


data.columns


# In[10]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# In[11]:


data = pd.read_csv("data funds.csv")


# In[12]:


data


# In[13]:


x = data.iloc[:,2:19]
x


# In[14]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[15]:


scaled_df = scaler.fit_transform(x)


# In[16]:


clusters_range = [2,3,4,5,6,7,8,9,10,11,12,13,14]
inertias = []

for c in clusters_range:
    kmeans = KMeans(init = "k-means++", n_clusters = c, n_init = 100, random_state = 0).fit(scaled_df)
    inertias.append(kmeans.inertia_)
plt.figure()
plt.plot(clusters_range, inertias, marker="o")
plt.show()


# In[36]:


kmeans = KMeans(3)
kmeans.fit(x)


# In[37]:


identified_clusters = kmeans.fit_predict(x)
identified_clusters


# In[43]:


data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['A'],data_with_clusters['D'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[46]:


data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['A'],data_with_clusters['J'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[47]:


data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['A'],data_with_clusters['G'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[48]:


data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['A'],data_with_clusters['M'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[49]:


data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['B'],data_with_clusters['E'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[50]:


data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['B'],data_with_clusters['H'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[51]:


data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['B'],data_with_clusters['K'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[52]:


data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['B'],data_with_clusters['N'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[53]:


plt.scatter(data_with_clusters['C'],data_with_clusters['F'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[54]:


plt.scatter(data_with_clusters['C'],data_with_clusters['I'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[55]:


plt.scatter(data_with_clusters['C'],data_with_clusters['L'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[56]:


plt.scatter(data_with_clusters['C'],data_with_clusters['O'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[ ]:





# In[17]:


plt.scatter(data_with_clusters['F'],data_with_clusters['G'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[18]:


plt.scatter(data_with_clusters['G'],data_with_clusters['H'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[19]:


plt.scatter(data_with_clusters['H'],data_with_clusters['I'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[20]:


plt.scatter(data_with_clusters['I'],data_with_clusters['J'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[21]:


plt.scatter(data_with_clusters['J'],data_with_clusters['K'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[22]:


plt.scatter(data_with_clusters['K'],data_with_clusters['L'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[23]:


plt.scatter(data_with_clusters['L'],data_with_clusters['M'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[24]:


plt.scatter(data_with_clusters['M'],data_with_clusters['N'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[25]:


plt.scatter(data_with_clusters['N'],data_with_clusters['O'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[ ]:




