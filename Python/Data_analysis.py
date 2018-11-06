#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Print all the interactive output without resorting to print, not only the last result
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# ### Aggeregation - agr and passing a function to it

# In[12]:


# https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
# UC irvine machine learning library

dftame_red_wine = pd.read_csv('winequality-red.csv', sep=';')
dftame_red_wine.head()


# In[7]:


# Average alcohol content
dftame_red_wine['alcohol'].mean()


# In[9]:


wino = dftame_red_wine.groupby('quality')
wino.describe()


# In[11]:


# define difference between max and min value of array
def max_to_min(arr):
    return arr.max() - arr.min()


# In[13]:


wino.agg(max_to_min)


# In[15]:


# Can pass string methods. This will yield same values as described in describe
wino.agg('mean')


# In[16]:


dftame_red_wine.head()


# In[19]:


# Add a new column
dftame_red_wine['qual/alc ratio'] = dftame_red_wine['quality'] / dftame_red_wine['alcohol']
dftame_red_wine.head()


# In[23]:


# Get exact same result as group by.
# wino = dftame_red_wine.groupby('quality')

wino_pivot = dftame_red_wine.pivot_table(index='quality')
wino_pivot.head()


# #### Plotting the data

# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')

dftame_red_wine.plot(kind='scatter', x='quality', y='alcohol')


# ### Split apply combine

# In[150]:


from IPython.display import Image
Image("split_apply_combine.png", width=400)

# Little example below


# In[128]:


# Example of group by and apply
dtest = pd.DataFrame({'A': 'c    a    a    b    c    b    a    c    c    b'.split(), 
                      'B': [1.1, 2.0, 3.2, 8.4, 8.8, 1.0, 0.1, 4.5, 2.1, 6.6], 
                      'Q': [4.1, 6.3, 5.9, 2.1, 4.1, 0,   2.3, 3.1, 8.6, 5.2]})
dtest


# Sort values with respect to Q
dtest.sort_values('Q', ascending=False, inplace=True)

# Now group. Each group will still have values sorted by Q
g = dtest.groupby('A')

# Shows groups
g.size()

def Rank(df):
    df['wrank'] = np.arange(len(df)) + 1
    return df

# Assign 'rank' array to the single group 
g.apply(Rank)

# dtest is printed....


# In[147]:


# Apply the result of experiment below 

dftame_red_wine = pd.read_csv('winequality-red.csv', sep=';')
dftame_red_wine.head(2)

# Sort by increasing Alcohol content
dftame_red_wine.sort_values('alcohol', ascending=False, inplace=True)
# Group
g = dftame_red_wine.groupby('quality')

# Apply Rank function
g.apply(Rank).head(3)

dftame_red_wine1 = g.apply(Rank)
dftame_red_wine1.head(3)

# Basically, the dataset is split into various groups, grouped by 'quality'. With in each group, it is sorted by  'alcohol' content.


# In[138]:


# Both should be same
g.size()
dftame_red_wine['quality'].value_counts()


# In[149]:


# Now print out from each quality class, wines with highest alcoholic content.

dftame_red_wine1.loc[dftame_red_wine1.wrank == 1]


# ### Easy tabulation

# In[164]:


import io
from io import StringIO

data= """
test_name  Name   IQ
A      Lynda  high
B      Sam    low
C      Ben    high
D      Bet    low
E      Bob    high
F      Tom    high
G      Sam    low
H      Ben    high
I      Bet    low
J      Bob    medium
K      Lynda  high
"""

dframe = pd.read_table(StringIO(data), sep='\s+')
dframe


# In[165]:


pd.crosstab(dframe.Name, dframe.IQ)
pd.crosstab(dframe.Name, dframe.IQ, margins=True)


# In[ ]:




