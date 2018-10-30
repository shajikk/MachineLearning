#!/usr/bin/env python
# coding: utf-8

# # Pandas - Series

# In[ ]:




import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# In[5]:


# Each value in the series in indexed
obj = Series([5, 10, 15, 20, 25])
obj


# In[9]:


obj.values
obj.index


# In[38]:


# Associative arrays 
ww2_cas = Series([8300000, 4300000, 3000000, 2100000, 400000], index = ['USSR', 'Germany', 'China', 'Japan', 'USA'])
ww2_cas.index
ww2_cas.values
ww2_cas


# In[14]:


ww2_cas['USA']


# In[17]:


# Bulk operations
ww2_cas[ww2_cas > 4000000]


# In[20]:


# key Exists
'USSR' in ww2_cas


# In[43]:


# Series to dictiionary
ww2_dict = ww2_cas.to_dict()
ww2_dict


# In[45]:


# From dict to Series
pd.Series(ww2_dict)


# In[51]:


# Now use the dict -> Series 
countries = ['USSR', 'Germany', 'China', 'Japan', 'USA', 'India']
pd.Series(ww2_dict, index=countries)


# In[50]:


countries = ['USSR']
pd.Series(ww2_dict, index=countries)


# In[77]:


# Check wheter a key is having/not having a null value

# Usage of pretty print
import pprint
pp = pprint.PrettyPrinter(indent=4)

print("\nPrinting Dict :\n")
pp.pprint(ww2_dict)

o = pd.Series(ww2_dict, index=countries)

print("\nisnull output :\n")
pp.pprint(pd.isnull(o))

print("\nnotnull output :\n")
pp.pprint(pd.notnull(o))


# In[84]:


s1 = Series([8300000, 4300000, 3000000, 2100000, 400000], index = ['USSR', 'Germany', 'China', 'Japan', 'USA'])
s2 = Series([1000000, 1000000, 1000000, 1000000, 100000], index = ['China', 'Japan', 'USA', 'USSR', 'Germany'])

# Stuff added up correctly even though the idexing is bot aligned
s3 = s1 + s2
s3


# In[90]:


# Explicit naming of series
s1.name = "Crap1"
s2.name = "Crap2"
s1.index.name = 'Countries'
s1


# # Pandas - DataFrame

# In[97]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# In[133]:


import webbrowser
website = 'https://en.wikipedia.org/wiki/List_of_all-time_NFL_win%E2%80%93loss_records'
#data = pd.read_html(website, header =0, flavor = 'bs4')
data = pd.read_html(website, header =0)
nfl_frame = data[1]
nfl_frame


# In[134]:


# Select First 10 Frames
nfl_frame = nfl_frame[1:10]

pp.pprint(nfl_frame.columns)
pp.pprint(nfl_frame.Team)


# In[135]:


# Example of index call
pp.pprint(nfl_frame['First NFL Season'])


# In[136]:


# Grab a list of columns
DataFrame(nfl_frame, columns=['Team', 'GP', 'First NFL Season' ])


# In[137]:


# Give a column that does not exist
DataFrame(nfl_frame, columns=['Team', 'GP', 'First NFL Season', 'City' ])


# In[144]:


# Select some head, tail
nfl_frame.head(2)
nfl_frame.tail(3)


# In[165]:


# Print Second row (Second - 1, as indexing start from index 0)

print(nfl_frame.iloc[2])
print("\n")

# Print Second column
print(nfl_frame.iloc[:,2])


# In[ ]:




