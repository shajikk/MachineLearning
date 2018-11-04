#!/usr/bin/env python
# coding: utf-8

# # Working with data

# In[18]:


# Print all the interactive output without resorting to print, not only the last result
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import webbrowser
website = 'https://en.wikipedia.org/wiki/List_of_all-time_NFL_win%E2%80%93loss_records'
#data = pd.read_html(website, header =0, flavor = 'bs4')
data = pd.read_html(website, header =0)
nfl_frame = data[1]
nfl_frame = nfl_frame.head()
nfl_frame


# In[19]:


# Export to CSV
nfl_frame.to_csv('nfl_info.csv', sep=',', index=False)


# In[20]:



import sys
nfl_frame.to_csv(sys.stdout, sep=',')


# In[21]:


pd.read_csv('nfl_info.csv')


# In[23]:


#Limit number of rows

pd.read_csv('nfl_info.csv', nrows=3)


# In[49]:


dframe = DataFrame(np.arange(9).reshape(3,3), index=['SF', 'LA', 'SD'], columns=['D1', 'D2', 'D3'])
dframe
dframe.to_csv('ex1.csv')

# Specify that 0 th col is the index column.
pd.read_csv('ex1.csv', index_col=0)


# In[55]:


# Selectively write a column out
dframe.to_csv('ex2.csv', columns=['D2'])
pd.read_csv('ex2.csv', index_col=0)


# ### Json object

# In[120]:


json_obj = """
{
    "name":"John",
    "age":30,
    "cars": [
    {
        "car1":"Ford",
        "car2":"BMW",
        "car3":"Fiat"
    },
    {
        "car1":"Toyota",
        "car2":"Nissan",
        "car3":"Tesla"
    }
    ]
 }
"""

import json
data = json.loads(json_obj)
data


# In[122]:


json.dumps(data)


# In[123]:


DataFrame(data['cars'])


# ### HTML support

# In[136]:


url = 'https://www.fdic.gov/bank/individual/failed/banklist.html'
dframe_list = pd.read_html(url)
dframe = dframe_list[0]
dframe.columns


# ### XLS file

# In[140]:


xlsfile = pd.ExcelFile('Sample.xlsx')


# In[141]:


dframe = xlsfile.parse('Sheet1')


# In[143]:


dframe.head()


# ### Data frame merge

# In[174]:


from random import shuffle
df1 = DataFrame({'key':list('ZABCDEFZ'), 'values1':[10, 0, 1, 2, 3, 4, 5, 88]})
df2 = DataFrame({'key':list('ZABCSZ'), 'values2':[11, 0, 1, 3, 8, 89]})
df1
df2
print("Common items are merged :")
pd.merge(df1,df2) # Same as pd.merge(df1, df2, on='key')


# In[175]:


# Choose any data frame and choose whose key columns to use
pd.merge(df1, df2, on='key', how='left')


# In[177]:


# Choose any data frame and choose whose key columns to use
pd.merge(df1, df2, on='key', how='right')


# In[178]:


# Union of both the data frames
pd.merge(df1, df2, on='key', how='outer')


# In[180]:


# Another example
df3 = DataFrame({'key':list('XXXYZZ'), 'values1':np.arange(6)})
df4 = DataFrame({'key':list('YYXXZ'), 'values2':np.arange(5)})
df3
df4


# In[185]:


pd.merge(df3, df4)


# ### Merge data frame by keys

# In[198]:


df_left = DataFrame({'key1' : ['Shaji', 'Roshni', 'Nida', 'Naina', 'Naina'], 
                     'key2' : [40, 37, 11, 3, 13],
                     'left_data' : ['good', 'good', 'bad', 'better', 'better']})
df_left
df_right = DataFrame({'key1' : ['Shaji', 'Shaji', 'Roshni', 'Nida', 'Naina', 'Naina'], 
                     'key2' : [40, 13, 11, 3, 44, 13],
                     'right_data' : ['good', 'bad', 'good', 'bad', 'better', 'better']})
df_right


# In[201]:


# Merge by two keys
pd.merge(df_left, df_right, on=['key1', 'key2'], how='outer')


# In[204]:


# Merge by one key. The other keys are automatically suffixed
pd.merge(df_left, df_right, on=['key1'], how='outer')


# In[207]:


# Merge by one key. The other keys are automatically suffixed, 
# Key suffixed explicitluy given.
pd.merge(df_left, df_right, on=['key1'], how='outer', suffixes=('__lefty', '__righty'))


# ### Merge/Join data frame by index

# In[211]:


df_left = DataFrame({'key1'  : ['X', 'Y', 'Z', 'X', 'Y'], 'data1' : range(5)})
df_right = DataFrame({'group_data' : [10, 11]}, index=['X', 'Y'])
df_left
df_right


# In[217]:


# Merge left frame using key1 and right frame using index
pd.merge(df_left, df_right, left_on='key1', right_index=True)


# In[230]:


df_left = DataFrame({'key1' : ['SF', 'SF', 'SF', 'LA', 'LA'],
                     'key2' : [10,    20,   30,   20,   30],
                    'data_set' : np.arange(5)})
df_left

df_right = DataFrame(np.arange(10).reshape(5,2),
                     index=[['LA', 'LA', 'SF', 'SF', 'SF'], 
                            [20,    10,   10,   10,   20]], columns=['col_1', 'col_2'])

df_right


# In[232]:


# Merges data frames with hierarchical index
pd.merge(df_left, df_right, left_on=['key1', 'key2'], right_index=True)


# In[238]:


df_left = DataFrame({'key1'  : ['X', 'Y', 'Z', 'X', 'Y'], 'data1' : range(5)})
df_right = DataFrame({'group_data' : [10, 11]}, index=['X', 'Y'])
df_left
df_right

# Default method of joining (MOSTLY using JOIN, use specific options in merge also as in join)

df_left.join(df_right)


# ### Concatinate Matrixes and data frames
# 

# In[244]:


m1 = np.arange(9).reshape(3,3)
m1
np.concatenate([m1,m1]) # same as np.concatenate([m1,m1], axis=0)
np.concatenate([m1,m1], axis=1)


# In[254]:


s1 = Series([1, 2, 3], index=['A', 'B', 'C'])
s1
s2 = Series([2, 3], index=['A', 'Q'])
s2
pd.concat([s1, s2])

# Hierarchically name series so that we can remembetr which one is which
pd.concat([s1, s2], keys=['s1_key', 's2_key'])


# In[256]:


# Series is converetd to a data frame
pd.concat([s1, s2], axis=1, sort=False)


# In[257]:


pd.concat([s1, s2], axis=1, sort=False, keys=['s1_key', 's2_key'])


# In[264]:


df1 = DataFrame(np.random.randn(4,3), columns=['X', 'Y', 'Z'])
df2 = DataFrame(np.random.randn(3,3), columns=['Z', 'A', 'X'])
df1
df2
pd.concat([df1, df2], sort=False)


# In[266]:


# To fix indexing
pd.concat([df1, df2], sort=False, ignore_index=True)


# In[ ]:




