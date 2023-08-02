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


# Display information about the DataFrame, including data types and non-null counts
df.info()


# In[6]:


# Count the number of missing values in each column of the DataFrame
df.isnull().sum()


# In[7]:


# Display the first 6 rows of the DataFrame
df.head(6)


# In[8]:


# Display the column names of the DataFrame
df.columns


# In[9]:


# Create a copy of the original DataFrame
data=df.copy()


# In[10]:


# Drop rows with missing values and store the result in a new DataFrame called 'data'
data=data.dropna()


# In[11]:


# Display the first few rows of the new DataFrame 'data'
data.head()


# In[12]:


# Verify that there are no missing values in the 'data' DataFrame
data.isnull().sum()


# In[13]:


import seaborn as sns
# Create a boxplot for the 'Modal Price(Rs./Quintal)' column to visualize its distribution and outliers
sns.boxplot(data['Modal Price(Rs./Quintal)'])


# In[14]:


# Display the shape (number of rows and columns) of the 'data' DataFrame
data.shape


# In[15]:


# Display the values of the 'Modal Price(Rs./Quintal)' column in the 'data' DataFrame
data['Modal Price(Rs./Quintal)']


# In[16]:


import plotly.express as px


# In[17]:


# Create a line plot using seaborn to visualize the relationship between 'District' and 'Modal Price(Rs./Quintal)'
# with different colors representing different 'Seasons'
sns.relplot(data=df,x="District",y="Modal Price(Rs./Quintal)",hue="Season",kind="line")


# In[18]:


# Create a bar plot using plotly express to visualize the 'Modal Price(Rs./Quintal)' for different 'Districts'
# with colors representing different 'Seasons'
fig=px.bar(df,x="District", y="Modal Price(Rs./Quintal)", color="Season",height=400)
fig.show()


# In[19]:


# Create a line plot using seaborn to visualize the relationship between 'Season' and 'Modal Price(Rs./Quintal)'
# with different colors representing different 'Seasons'
sns.relplot(data=df,x="Season",y="Modal Price(Rs./Quintal)",hue="Season",kind="line")


# In[20]:


# Create a line plot using seaborn to visualize the relationship between 'Day' and 'Modal Price(Rs./Quintal)'
# with different colors representing different 'Seasons'
sns.relplot(data=df,x="Day",y="Modal Price(Rs./Quintal)",hue="Season",kind="line")


# In[21]:


# Create a line plot using seaborn to visualize the relationship between 'Month' and 'Modal Price(Rs./Quintal)'
# with different colors representing different 'Seasons'
sns.relplot(data=df,x="Month",y="Modal Price(Rs./Quintal)",hue="Season",kind="line")


# In[22]:


# Display information about the 'data' DataFrame, including data types and non-null counts
data.info()


# In[23]:


# The following section of code seems to be converting categorical data into numerical data using mapping.
# Similar mapping for other categorical columns 'Varieties', 'Season', 'Growth', 'Harvest', 'Market', 'District', 'Month'
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
# Create a heatmap to visualize the correlation between numerical features in the 'data' DataFrame
dataplot=sns.heatmap(data.corr(),cmap="YlGnBu",annot=True)
plt.show()


# In[34]:


# Display the column names of the 'data' DataFrame
data.columns


# In[35]:


# The following section seems to be preparing the data for training a machine learning model.
# Separate the features and labels from the 'data' DataFrame
features=data[['Crops', 'Varieties', 'Season', 'Growth', 'Harvest', 'Market',
       'District', 'Day', 'Month', 'Year']]
labels=data['Modal Price(Rs./Quintal)']


# In[36]:


# Split the data into training and testing sets
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
# Initialize and train a K-Nearest Neighbors Regressor model with k=5
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(Xtrain, Ytrain)


# In[39]:


# Make predictions using the trained KNN Regressor model on the training data
y_pred2=knn.predict(Xtrain)


# In[40]:


y_pred2


# In[41]:


from sklearn.metrics import r2_score


# In[42]:


# Calculate the accuracy to evaluate the performance of the KNN model
r2_score(Ytrain,y_pred2)


# In[43]:


d={'true':Ytrain,'predicted':y_pred2}
drf_df=pd.DataFrame(data=d)
drf_df['diff']=drf_df['predicted']-drf_df['true']
print(drf_df)


# In[ ]:




