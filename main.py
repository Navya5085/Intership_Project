#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary libraries
import numpy as np


# In[2]:


import seaborn as sns


# In[3]:


import pandas as pd


# In[4]:


# Read the dataset from the CSV file into a pandas DataFrame
df=pd.read_csv('Crop dataset (5).csv',na_values='=')
df


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.head(6)


# In[8]:


df.columns


# In[9]:


data=df.copy()


# In[10]:


data=data.dropna()


# In[11]:


data.head()


# In[12]:


data.isnull().sum()


# In[13]:


import seaborn as sns
sns.boxplot(data['Modal Price(Rs./Quintal)'])


# In[14]:


data.shape


# In[15]:


data['Modal Price(Rs./Quintal)']


# In[16]:


import plotly.express as px


# In[17]:


sns.relplot(data=df,x="District",y="Modal Price(Rs./Quintal)",hue="Season",kind="line")


# In[18]:


fig=px.bar(df,x="District", y="Modal Price(Rs./Quintal)", color="Season",height=400)
fig.show()


# In[19]:


sns.relplot(data=df,x="Season",y="Modal Price(Rs./Quintal)",hue="Season",kind="line")


# In[20]:


sns.relplot(data=df,x="Day",y="Modal Price(Rs./Quintal)",hue="Season",kind="line")


# In[21]:


sns.relplot(data=df,x="Month",y="Modal Price(Rs./Quintal)",hue="Season",kind="line")


# In[22]:


data.info()


# In[23]:


dist=(data['Crops'])
distset=set(dist)
dd=list(distset)
dict0fWords={dd[i]: i for i in range(0,len(dd))}
data['Crops']=data['Crops'].map(dict0fWords)


# In[24]:


dist=(data['Varieties'])
distset=set(dist)
dd=list(distset)
dict0fWords={dd[i]: i for i in range(0,len(dd))}
data['Varieties']=data['Varieties'].map(dict0fWords)


# In[25]:


dist=(data['Season'])
distset=set(dist)
dd=list(distset)
dict0fWords={dd[i]: i for i in range(0,len(dd))}
data['Season']=data['Season'].map(dict0fWords)


# In[26]:


dist=(data['Growth'])
distset=set(dist)
dd=list(distset)
dict0fWords={dd[i]: i for i in range(0,len(dd))}
data['Growth']=data['Growth'].map(dict0fWords)


# In[27]:


dist=(data['Harvest'])
distset=set(dist)
dd=list(distset)
dict0fWords={dd[i]: i for i in range(0,len(dd))}
data['Harvest']=data['Harvest'].map(dict0fWords)


# In[28]:


dist=(data['Market'])
distset=set(dist)
dd=list(distset)
dict0fWords={dd[i]: i for i in range(0,len(dd))}
data['Market']=data['Market'].map(dict0fWords)


# In[29]:


dist=(data['District'])
distset=set(dist)
dd=list(distset)
dict0fWords={dd[i]: i for i in range(0,len(dd))}
data['District']=data['District'].map(dict0fWords)


# In[30]:


dist=(data['Month'])
distset=set(dist)
dd=list(distset)
dict0fWords={dd[i]: i for i in range(0,len(dd))}
data['Month']=data['Month'].map(dict0fWords)


# In[31]:


data.info()


# In[32]:


import matplotlib.pyplot as plt


# In[33]:


#coorelation heatmap
dataplot=sns.heatmap(data.corr(),cmap="YlGnBu",annot=True)
plt.show()


# In[34]:


data.columns


# In[35]:


features=data[['Crops', 'Varieties', 'Season', 'Growth', 'Harvest', 'Market',
       'District', 'Day', 'Month', 'Year']]
labels=data['Modal Price(Rs./Quintal)']


# In[36]:


from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(features,labels,test_size=0.2,random_state=2)


# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree


# In[38]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(Xtrain, Ytrain)


# In[39]:


y_pred2=knn.predict(Xtest)


# In[40]:


y_pred2


# In[41]:


from sklearn.metrics import r2_score


# In[42]:


r2_score(Ytest,y_pred2)


# In[43]:


d={'true':Ytest,'predicted':y_pred2}
drf_df=pd.DataFrame(data=d)
drf_df['diff']=drf_df['predicted']-drf_df['true']
print(drf_df)


# In[ ]:




