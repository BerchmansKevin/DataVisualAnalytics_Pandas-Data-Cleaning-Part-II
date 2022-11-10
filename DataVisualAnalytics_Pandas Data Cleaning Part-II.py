#!/usr/bin/env python
# coding: utf-8

# # `BERCHMANS KEVIN S`
# 
# 

# # `Pandas Data Cleaning Part-II`
# 
# ## `LabelEncoder in Scikit Learn`
# 
# Encodes string values as integer values

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder


# In[2]:


le = LabelEncoder()

# New object
df = pd.DataFrame(data = {'col1': ['foo','bar','foo','bar'],
                          'col2': ['x', 'y', 'x', 'z'],
                          'col3': [1, 2, 3, 4]})


# In[3]:


# Now convert string values of each column into integer values
df.apply(le.fit_transform)


# ### `One Hot Encoder`
# 
# Consider the following dataframe. You will have to represent string values of column A and B with integers

# In[4]:


df = pd.DataFrame({'A': ['a','b','a'], 'B': ['b','a','c'], 'c': [1,2,3]})
df


# In[5]:


# Call get_dummies method. It will create a new column for each string values in DF columns

pd.get_dummies(df, prefix=['col1', 'col2'])


# ### `MinMaxScaler`
# 
# It will transform values into a range of 0 to 1

# In[6]:


from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler(feature_range=(0,1))  #(0,1) is default range

df2 = pd.DataFrame({"col1": [5,-41,-67],
                    "col2": [23,-53,-36],
                   "col3": [-25,10,17]})
mm_scaler.fit_transform(df2)


# ### `Binarizer `
# It will encode values into 0 or 1, depending on the threshold

# In[7]:


from sklearn.preprocessing import Binarizer

dfb = pd.DataFrame({"col1": [110,200],
                  "col2": [120,800],
                  "col3": [310,400]})

bin = Binarizer(threshold=300)
bin.fit_transform(dfb)


# ### `Imputer`
# 
# You can also use Imputer from sklearn to handle NaN objects in each columns. Here, we replace NaN with
# column mean value. This is good alternative to fillna() method.

# In[8]:


import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
df = pd.DataFrame({"col1": [7,2,3],
                  "col2": [4,np.nan,6],
                  "col3": [np.nan,np.nan,3],
                  "col4": [10,np.nan,9]})
print(df)
imp_mean.fit_transform(df)


# # `De-duplication or Entity Resolution and String Matching`
# 
# You can use dedupe and fuzzywuzzy packages. Install them using pip3 and import inside your Python code

# `Conclusion :` Life is not just a bunch of Kaggle datasets, where in reality you’ll have to make
# decisions on how to access and clean the data you need everyday. Sometimes you’ll have a
# lot of time to make sure everything is in the right place, but most of the time you’ll be pressed
# for answers. If you have the right tools in place and understanding of what is possible, you’ll
# be able to get to those answers easily.
