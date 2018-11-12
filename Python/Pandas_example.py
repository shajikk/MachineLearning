#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Print all the interactive output without resorting to print, not only the last result
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# # Pandas - Series

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


# In[170]:


import webbrowser
website = 'https://en.wikipedia.org/wiki/List_of_all-time_NFL_win%E2%80%93loss_records'
#data = pd.read_html(website, header =0, flavor = 'bs4')
data = pd.read_html(website, header =0)
nfl_frame = data[1]
nfl_frame


# In[171]:


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


# In[177]:


# Change only one column
nfl_frame_copy  = nfl_frame.copy()
nfl_frame_copy['Division'] = "PPP unknown"
nfl_frame_copy


# In[180]:


# Change only one column another example
nfl_frame_copy  = nfl_frame.copy()
nfl_frame_copy['Division'] = np.arange(9)
nfl_frame_copy


# In[187]:


s = pd.Series(["two", "three", "four"], index=[2,3,4])
pp.pprint(s)

nfl_frame_copy['Division'] = s
nfl_frame_copy


# In[189]:


# Delete a column
nfl_frame_copy1 = nfl_frame_copy.copy()
del nfl_frame_copy1['Division']
nfl_frame_copy1

# Convert a dictionary to python
d = {'Visa' : ['H1b', 'L1', 'F1', 'B2', 'O'], 'Type' : ['professional', 'transfer', 'Student', 'Visitor', 'Genius']}
visa_frame = DataFrame(d)
visa_frame
# ### DataFrame to dictionary

# In[197]:


visa_frame.to_dict() # Same as : visa_frame.to_dict('dict')


# In[199]:


visa_frame.to_dict('list') # Other options are 'series', 'records', 'split, 'index'


# # Index Objects

# In[210]:


# Create a series 
my_series = Series([1, 2, 3, 4, 5], index=['A', 'B', 'C', 'D', 'E'])
my_index  = my_series.index
pp.pprint(my_index[2:])
pp.pprint(my_index[2])


# In[218]:


# Deals with missing data points..
my_series1 = my_series
my_series2 = my_series1.reindex(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
print("Deals with Missing stuff:\n")
pp.pprint(my_series2)
print("\n")


my_series3 = my_series2.reindex(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], fill_value='KKKK')
my_series3


# In[228]:


my_countries = Series(['USA', 'UK', 'India'], index=[0, 5, 10])
# Fill up as per the index above
my_countries.reindex(range(20), method='ffill')


# ### Reindex data frame

# In[270]:


dframe = DataFrame(np.random.randn(25).reshape(5,5))
print("\nReshape output")
dframe
print("\nReshape with label added output")
dframe = DataFrame(np.random.randn(25).reshape(5,5), index=['A', 'B', 'D', 'E', 'F'], columns=['C1', 'C2', 'C3', 'C4', 'C5' ])
dframe
print("\nInserts new rows / colmns output")
# Inserts new rows / colmns
dframe.reindex(index=['A', 'B', 'C', 'D', 'E', 'F'], columns=['C1', 'C2', 'C3', 'C31', 'C4', 'C5' ])

# Inserts new rows / colmns
print("\nSelects range of labelled indexes, note that 'C' is not defined")
dframe.loc['A':'D']


# ### Drop an entry

# In[279]:


dframe = DataFrame(np.arange(9).reshape(3,3), index=['SF', 'LA', 'SD'], columns=['D1', 'D2', 'D3'])
dframe
dframe1 = dframe
dframe.drop('LA')


# In[280]:


# Drop a column

dframe1.drop('D2', axis=1)


# ### Select entry and replace

# In[300]:


dframe = DataFrame(np.arange(9).reshape(3,3), index=['SF', 'LA', 'SD'], columns=['D1', 'D2', 'D3'])
dframe = dframe * 2
dframe
dframe1 = DataFrame(np.arange(9).reshape(3,3))
dframe1 = dframe1 * 2
dframe1


# In[371]:


dframe = DataFrame(np.arange(9).reshape(3,3), index=['SF', 'LA', 'SD'], columns=['D1', 'D2', 'D3'])
dframe = dframe * 2
dframe1 = DataFrame(np.arange(9).reshape(3,3))
dframe1 = dframe1 * 2

dframe.loc['SF':'SD']
dframe1.iloc[0:3]      # 0 to 3-1 indexes


print("\nReplace based on loc\n")

dframe.loc[dframe.D1 >2]

dframe.loc[dframe.D1 > 2, 'D3'] = 5000
dframe


# In[381]:


dframe = DataFrame(np.arange(9).reshape(3,3), index=['SF', 'LA', 'SD'], columns=['D1', 'D2', 'D3'])
dframe = dframe * 2
dframe
dframe > 10

# Grab by logic
print("\nGrab by logic\n")

dframe[dframe>10]
dframe.where(dframe>10)


# In[3]:


d = pd.DataFrame({'year':[2008,2008,2008,2008,2009,2009,2009,2009], 
                   'flavour':['strawberry','strawberry','banana','banana', 'strawberry','strawberry','banana','banana'],
                  'day':['sat','sun','sat','sun','sat','sun','sat','sun'],
                   'sales':[10,12,22,23,11,13,23,24]})

d

# 'sales' gets replaced
d.loc[(d.day== 'sun') & (d.flavour== 'banana') & (d.year== 2009),'sales'] = 10000
d.loc[d.day == 'sun']


# ### Data Alignment

# In[387]:


# Add two data Series

ser1 = Series([1, 2, 3, 4, 5], index=['A', 'B', 'C', 'D', 'E'])
ser2 = Series([2, 2, 1, 2], index=['A', 'B', 'C', 'D'])
ser3 = ser1 + ser2
ser3


# In[397]:


# Addition of data frames
df1 = DataFrame(np.arange(4).reshape(2,2), columns=list('AB'), index=list('PQ'))
df1
df2 = DataFrame(np.arange(9).reshape(3,3), columns=list('ABC'), index=list('PQR'))
df2
df3 = df1+df2
df3


# In[399]:


df1.add(df2, fill_value=0)


# In[408]:


# Extract a series out of d frame and substract.
ser = df2.loc['P']
df2 - ser


# ### Sort stuff

# In[418]:


ser1 = Series([1, 2, 3, 4, 5], index=list('DSMAC'))
ser1.sort_index()
ser1.sort_values()


# In[419]:


ser2 = Series(np.random.randn(10))
ser2
ser2.sort_values()
# Rank is Preset. 
ser2.rank()


# ### Summary Statistics

# In[449]:


df = DataFrame(np.arange(16).reshape(4,4), columns=list('ABCD'), index=list('PQRS'))
df.loc['P','D'] = np.nan
df.loc['R','A'] = np.nan
df

df.sum()  # col
df.sum(axis=1) # row

df.min() # col
df.idxmin() # col

df.min(axis=1) # row
df.idxmin(axis=1) #row

df.cumsum()


# In[452]:


# describe

df = DataFrame(np.arange(16).reshape(4,4), columns=list('ABCD'), index=list('PQRS'))
df.loc['P','D'] = np.nan
df.loc['R','A'] = np.nan
df
df.describe()


# # Corelation and covariance

# In[455]:


from pandas_datareader import data as pdweb
import datetime


# In[462]:


prices = pdweb.get_data_yahoo(['CVX', 'XOM', 'BP'], start=datetime.datetime(2015,1,1), end=datetime.datetime(2017,1,1))['Adj Close']
prices.head()
volume = pdweb.get_data_yahoo(['CVX', 'XOM', 'BP'], start=datetime.datetime(2015,1,1), end=datetime.datetime(2017,1,1))['Volume']
volume.head()


# In[472]:


get_ipython().run_line_magic('matplotlib', 'inline')
prices.plot()


# In[530]:


import seaborn as sns
import matplotlib.pyplot as plt

#   Returns is % change in prices 
rets = prices.pct_change()
rets.head()

print("Correlation between different tickers. Diagonal is 1")
corr = rets.corr()
corr

print("Heatmap")

sns.heatmap(corr, annot=True)


# #### Example of correlation matrix

# In[521]:


Index= ['aaa', 'bbb', 'ccc', 'ddd', 'eee']
Cols = ['A', 'B', 'C', 'D']
#np.arange(20).reshape(5,4)
df = DataFrame(np.arange(20).reshape(5,4), index=Index, columns=Cols)
df

# Correlation between A and B
df['A'].corr(df['B'])

# Correlation to itself.
df.corr()


# #### Example of heatmap, with seaborn

# In[491]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[508]:


#colormap = sns.diverging_palette(220, 10, as_cmap=True)
#sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
get_ipython().run_line_magic('matplotlib', 'inline')
Index= ['aaa', 'bbb', 'ccc', 'ddd', 'eee']
Cols = ['A', 'B', 'C', 'D']
df = DataFrame(abs(np.random.randn(5, 4)), index=Index, columns=Cols)
sns.heatmap(df, annot=True)


# ### Other operations

# In[536]:


ser1 = Series(['a', 'b', 'c', 'd', 'a'])
ser1.unique()
ser1.value_counts()


# In[543]:


s1 = Series(['a', 'b', np.nan, 'd', np.nan])
s1.isnull()

# Drop null
s1.dropna()


# In[551]:


df = DataFrame(np.arange(16).reshape(4,4), columns=list('ABCD'), index=list('PQRS'))
df.loc['P','A'] = np.nan
df.loc['P','B'] = np.nan
df.loc['P','C'] = np.nan
df.loc['P','D'] = np.nan
df.loc['R','A'] = np.nan
df

# Drop null rows
df.dropna()

#Drop those rows where all values are null
df.dropna(how='all')

# Drop null rows, here every col is dropped
df.dropna(axis=1)


# In[564]:


df = DataFrame(np.arange(16).reshape(4,4), columns=list('ABCD'), index=list('PQRS'))
df.loc['P','A'] = np.nan
df.loc['P','B'] = np.nan
df.loc['P','C'] = np.nan
df.loc['P','D'] = np.nan

df.loc['Q','A'] = np.nan
df.loc['Q','B'] = np.nan
df.loc['Q','C'] = np.nan


df.loc['R','A'] = np.nan
df.loc['R','B'] = np.nan

df.loc['S','A'] = np.nan

df

# Drop all with threshold 2 and below
df.dropna(thresh=2)
df.dropna(thresh=3)

# All na-s are filled with given values
df.fillna(100)
df.fillna('a')

# Changed in place
df.fillna({'A':'a', 'B':'b', 'C':'c','D':'d'}, inplace=True)
df


# # Index hierarchy

# In[579]:


ser1 = Series(np.random.randn(6), index=[[1, 1, 1, 2, 2, 2], ['a', 'b', 'c', 'a', 'b', 'c']])
ser1


# In[580]:


ser1.index


# In[581]:


ser1[1]


# In[582]:


ser1[1][0]
ser1[1][1]
ser1[1][2]


# In[584]:


# All 'a's in lower index level
ser1[:,'a']


# In[586]:


dframe = ser1.unstack()
dframe


# #### Index hierarchy in data frame

# In[600]:


dframe = DataFrame(np.arange(16).reshape(4, 4), index=[ ['a', 'a', 'b', 'b'], [1,2, 1, 2]], columns=[ ['SF', 'SF', 'NY', 'NY'], ['Hot', 'Cold', 'Hot', 'Cold']])
dframe

# name the Indexes
dframe.index.names = ['INDEX1', 'INDEX2']
dframe.columns.names = ['City', 'Temp']
dframe


# In[604]:


# Swap levels
dframe.swaplevel('City', 'Temp', axis=1)


# In[611]:


# Sort by level 0 index
dframe.sort_index(level=0)


# In[610]:


#sort by level 1 index
dframe.sort_index(level=1)


# In[614]:


dframe.sum()
dframe.sum(level='Temp', axis=1)


# In[ ]:




