#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


path = '/Users/QXJ/Desktop/IBM/Analysis data with python/archive/kc_house_data.csv'
df = pd.read_csv(path)


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


df.drop(columns=['id','date'], axis = 1, inplace = True)


# In[6]:


df.head()


# In[7]:


df.describe()


# In[8]:


df.dtypes


# In[9]:


df.info()


# In[10]:


missing_data = df.isnull()
missing_data.head()


# In[11]:


df_bedroom = df[['bedrooms']].value_counts()
df_bedroom


# In[12]:


x = df['bedrooms']
y =  df['price']
plt.scatter(x,y)
plt.show()


# In[13]:


df.drop(df[df['bedrooms']==33].index, axis = 0, inplace = True)


# In[14]:


x = df['bedrooms']
y =  df['price']
plt.scatter(x,y)
plt.show()


# In[15]:


df[['bathrooms']].value_counts()


# In[16]:


df_floors = df[['floors']].value_counts()


# In[17]:


df_floors = df_floors.to_frame()
df_floors


# In[18]:


df_floors.columns


# In[19]:


df_floors.reset_index(inplace=True)


# In[20]:


df_floors


# In[21]:


df_floors.columns


# In[22]:


df_floors.columns = ['floors', 'number of apartments']
df_floors


# In[23]:


df['waterfront'].unique()


# In[24]:


sns.set(rc={"figure.figsize":(12, 6)})
sns.set_style("whitegrid", {'axes.grid' : False})
ax = sns.boxplot(x='waterfront', y = 'price', data = df)
ax.set(xlabel='waterfront', ylabel='price in million',  title ='The impact view on apartment price' )


# In[25]:


ax = sns.regplot(x='sqft_above', y = 'price', data = df, color = 'g')
ax.set(xlabel='sqft_above', ylabel='price in million',  title ='The correlation between sqrt_above and price' )


# In[26]:


#Fit a linear regression model to predict
import sklearn
from sklearn.linear_model import LinearRegression


# In[31]:


lm = LinearRegression()
lm


# In[32]:


x = df[['sqft_living']]
y = df['price']
lm.fit(x, y)


# In[40]:


R_square = round(lm.score(x,y),3)
print('R_square = ' + str(R_square))


# In[43]:


x_m = df[['floors', 'waterfront', 'lat', 'bedrooms', 'sqft_basement', 'view', 'bathrooms', 'sqft_living15', 'sqft_above','grade','sqft_living']]
y = df['price']
lm.fit(x_m,y)


# In[44]:


lm.score(x_m,y)


# ## Building pipeline

# In[72]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


# In[73]:


Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]


# In[74]:


pipe = Pipeline(Input)


# In[75]:


z = x_m.astype(float)
pipe.fit(z,y)


# In[76]:


yhat = pipe.predict(z)
yhat[0:4]


# In[77]:


pipe.score(z,y)


# ## Ridge regression

# In[78]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


# In[79]:


RigeModel = Ridge(alpha = 1)


# In[86]:


x_train, x_test, y_train, y_test = train_test_split(x_m, y, test_size = 0.1, random_state = 0)
print('number of test samples:', x_test.shape[0])
print('number of training samples:',x_train.shape[0])


# In[87]:


RigeModel.fit(x_train,y_train)
RigeModel.score(x_test, y_test)


# In[89]:


yhat = RigeModel.predict(x_test)
yhat[0:4]


# In[90]:


ax1 = sns.kdeplot(y_test, color = 'r', label = 'acutal values')
sns.kdeplot(yhat, color = 'b', label = 'fitted values', ax = ax1)
ax1.set_title('actual vs. fitted prices')
plt.show()


# In[92]:


pr = PolynomialFeatures(degree =3)
x_pr = pr.fit_transform(x_m)


# In[94]:


x_train, x_test, y_train, y_test = train_test_split(x_pr, y, test_size = 0.1, random_state = 0)
print('number of test samples:', x_test.shape[0])
print('number of training samples:',x_train.shape[0])


# In[95]:


RigeModel.fit(x_train,y_train)
RigeModel.score(x_test,y_test)


# In[96]:


yhat = RigeModel.predict(x_test)
yhat[0:4]


# In[97]:


ax1 = sns.kdeplot(y_test, color = 'r', label = 'acutal values')
sns.kdeplot(yhat, color = 'b', label = 'fitted values', ax = ax1)
ax1.set_title('actual vs. fitted prices')
plt.show()


# In[ ]:




