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


# ### Combine data frames

# In[274]:


s1 = Series([2, np.nan, 3, np.nan, 4, np.nan], index=list('PQRSTU'))
s1
len(s1)
s2 = Series(np.arange(len(s1)), dtype=np.float64, index=list('PQRSTU'))
s2


# In[278]:


# Create a new series where Values from Series 2 is chosen if Corresponding value in Series 2 is not defined or null.
Series(np.where(pd.isnull(s1), s2, s1), index= s1.index)

# Same stuff below
s1.combine_first(s2)


# In[285]:


nan = np.nan
dframe_odds = DataFrame({'X' : [1.0, nan, 3.0, nan],
                         'Y' : [nan, 3.0, nan, 5.0],
                         'Z' : [nan, 4.0, nan, 7.0]})

dframe_even = DataFrame({'X' : [nan, nan, 4.0, 10.0, 12],
                         'Y' : [nan, 3.3, 4.0, 5.0,  13]})

dframe_odds
dframe_even


# In[286]:


dframe_odds.combine_first(dframe_even)


# ### Reshaping

# In[313]:


df1 = DataFrame(np.arange(8).reshape(2,4), index=['LA', 'SF'], columns=['A', 'B', 'C', 'D'])
df1

# Note below, name index & columns specifically
df1 = DataFrame(np.arange(8).reshape(2,4), index=pd.Index(['LA', 'SF'], name='city'), 
                                           columns=pd.Index(['A', 'B', 'C', 'D'], name='class')
               )
df1


# In[302]:


# Convert above to stack 
df2 = df1.stack()
df2


# ### unstack stuff convert Series to data frame

# In[311]:


# Different ways of unstacking

orig1 = df2.unstack() # Default
orig1

orig2 = df2.unstack('class')
orig2

orig3 = df2.unstack('city')
orig3


# In[ ]:


# Another example
s1 = Series([0, 1, 3], index=['A', 'B', 'C'])
s2 = Series([4, 5, 6], index=['X', 'Y', 'Z'])


# In[320]:


dframe = pd.concat([s1, s2], keys=['alpha', 'beta'])
dframe
dframe.unstack()


# In[324]:


# Got rid off Null values
dframe.unstack().stack()


# In[326]:


# Keep Null values during stacking
dframe.unstack().stack(dropna=False)


# ### Pivot tables and manipulation

# In[328]:


import pandas.util.testing as tm; tm.N = 3
def unpivot(frame):
    N, K = frame.shape
    data = {'value' : frame.values.ravel('F'),
            'variable' : np.asarray(frame.columns).repeat(N),
            'date' : np.tile(np.asarray(frame.index), K)}
    return pd.DataFrame(data, columns=['date', 'variable', 'value'])
df = unpivot(tm.makeTimeDataFrame())


# In[329]:


df


# In[333]:


dframe_p = df.pivot('date', 'variable', 'value')
dframe_p


# In[334]:


dframe_p = df.pivot('variable', 'date', 'value')
dframe_p


# ### Finding duplicates

# In[340]:


df = DataFrame({'key1' : ['A'] * 3 + ['B'] * 2, 
               'key2' :  [1, 1, 2, 2, 2]})
df


# In[341]:


df.duplicated()


# In[342]:


df.drop_duplicates()


# In[345]:


# Duplicate looking at only key1, ignore other column
df.drop_duplicates('key1')


# In[362]:


# Keep the last one
df.drop_duplicates(['key1'], keep='last')


# ### Mapping

# In[366]:


dframe = DataFrame({ 'city' : ['Denver', 'Tahoe', 'Fox Park'], 
                   'altitude' : [5000, 7000, 6000]})
dframe

state_map = {'Denver' : 'CO', 'Tahoe' : 'CA', 'Fox Park' : 'WY'}


# In[371]:


dframe['state'] = dframe['city'].map(state_map)
dframe


# ### Replace

# In[372]:


s1 = Series([1, 2, 3, 4, 5, 6])
s1


# In[377]:


# Different ways of replacement
s1.replace(2, 100)
s1.replace([1,3,4], [100, 300, np.nan])
s1.replace({1:100, 3:300, 4:np.nan})


# ### Rename Index

# In[379]:


df = DataFrame(np.arange(12).reshape(3,4), index=['NY', 'SF', 'LA'], columns=['A', 'B', 'C', 'D'])
df


# In[385]:


df.index.map(str.lower)
df.index = df.index.map(str.lower)
df


# In[387]:


# Another way to change index
df.rename(index=str.title, columns=str.lower)


# In[392]:


# Rename index using dictionary
df.rename(index={'ny' : 'New York'}, columns={'A' : 'Col A'})


# In[394]:


# The effects are permanent 
df.rename(index={'ny' : 'New York'}, columns={'A' : 'Col A'}, inplace=True)
df


# ### Binning

# In[435]:


years = [1920, 1945, 1947, 1932, 1950, 1952, 1960, 1970, 1978, 1982, 1984, 1994, 1996, 2000, 2007, 2008, 2010, 2015, 2018]

# Bins are specifically given
decade_bins = [1960, 1970, 1980, 1990, 2000, 2010, 2020]


# In[432]:


# Usage of cut during binning
decade_cat = pd.cut(years, decade_bins)
decade_cat


# In[433]:


# Various categories
decade_cat.categories

# Count of values after catergorizing
pd.value_counts(decade_cat)


# In[434]:


# All the years have been seperated to two bins
years = [1920, 1945, 1947, 1932, 1950, 1952, 1960, 1970, 1978, 1982, 1984, 1994, 1996, 2000, 2007, 2008, 2010, 2018]
cat1 = pd.cut(years, 2, precision=10)
pd.value_counts(cat1)


# ### Outliers

# In[487]:


# Example to mask and change

p = pd.DataFrame({('A','a'): [-1,-1,0,10,12],
                   ('A','b'): [0,1,2,3,-1],
                   ('B','a'): [-20,-10,0,10,20],
                   ('B','b'): [-200,-100,0,100,200]})
p
mask = p.loc[:]<0
p[mask] = 1000
p


# In[488]:


# Set the seed
np.random.seed(121)
dframe = DataFrame(np.random.randn(1000, 5))
dframe.tail(5)
dframe.describe()


# In[489]:


col_0 = dframe[0]
col_0.tail()

# Show me the values of this Series that is > 3
col_0[np.abs(col_0) > 3]


# In[490]:


# Do same to the entire frame
dframe[(np.abs(dframe) > 3).any(1)].head()


# In[497]:


# Cap such  values to 3. Application of loc

#mask = p.loc[:]<0
#p[mask] = 1000

# Get True / false array
mask = np.abs(dframe[:]) > 3

dframe[mask] = np.sign(dframe)*3

# Print the 4-th row 
dframe.loc[4,:]


dframe.describe()


# ### Permutation

# In[506]:


blender = np.random.permutation(4)
blender


# In[507]:


dframe = DataFrame(np.arange(16).reshape(4,4))
dframe


# In[508]:


# Usage of take method, it changed the indexing as per the blender
dframe.take(blender)


# In[516]:


# Create an array of random inregers between 0 and 3 of size 10
shaker = np.random.randint(0, 4, size=10)
shaker


# ### Group by on dataframes

# In[520]:


dframe = DataFrame({'key1' : ['A', 'B', 'A', 'C', 'B'],
                    'key2'  : ['alpha', 'beta', 'beta', 'alpha', 'beta'], 
                    'val1' : np.random.randn(5),
                    'val2' : np.random.randn(5)})
dframe


# In[526]:


group1 = dframe['val1'].groupby(dframe['key1'])


# In[560]:


df = pd.DataFrame()
pet_list = ['cat', 'hamster', 'alligator', 'snake']

is_mammal = {'cat' : 'mammal', 'hamster' : 'mammal', 'alligator' : 'reptile', 'snake' : 'reptile' }


# In[573]:



pet = [np.random.choice(pet_list) for i in range(1,15)]
        
# Random weight of animal column
weight = [np.random.choice(range(5,15)) for i in range(1,15)]
        
# Random length of animals column
length = [np.random.choice(range(1,10)) for i in range(1,15)]
        
# random age of the animals column
age = [np.random.choice(range(1,15)) for i in range(1,15)]

df['animal'] = pet
df['age']    = age
df['weight'] = weight
df['length'] = length

df['type'] = df['animal'].map(is_mammal)

df


#The above data is classified into various animal buckets. Further operations can be done on it.
animal_groups = df.groupby("animal")
animal_groups['weight'].mean()

from IPython.display import Image
Image("xZnMuPZ.jpg", width=700)


# In[576]:


# This shows age, weight and length together.
df.groupby("animal").mean()

# Another grouping, shows age, weight and length
df.groupby("type").mean()


# In[578]:


# Show more data, merge the operations above and show it in a single table
df.groupby(['type','animal']).mean()


# In[634]:


# Group by 'type' and 'animal' and select the age from the bin. Find means of all ages.
df.groupby(['type','animal'])['age'].mean()


# In[631]:


# show size
df.groupby(['type','animal']).size()


# In[597]:


# Iterate over each groups/bins created.
for name, group in df.groupby('animal'):
    print("Animal Name >>> " + name + '\n')
    print(group)
    print ("\n")


# In[603]:


# Another type of iterating, just as above. Shows type as well
df.head()
for (keys, group) in df.groupby(['animal', 'type']):
    print("k1 = %s, k2 = %s\n" %(keys[0], keys[1]))
    print(group)
    print('\n')


# In[629]:


# Array to dict convertion
t1 = [('a', 'b'), ('c', 'd')]
dict(t1)

sample_dict = dict(list(df.groupby('animal')))
sample_dict

sample_dict['alligator']

# Seperate the numbers Vs data types, use below
sample_dict = dict(list(df.groupby(df.dtypes, axis=1)))
sample_dict


# ### Misc operations, organize by sub index. Select using loc

# In[700]:


animals = DataFrame(np.random.randint(0, 100, size=30).reshape(6,5), 
                    index=[['large', 'large', 'large', 'large', 'small', 'small'],
                           ['Cat', 'Dog', 'Lion', 'Cow', 'Ant', 'Mouse']],
                    columns=['A', 'B', 'C', 'D', 'E'])
animals


# In[711]:


animals = DataFrame(np.random.randint(0, 100, size=30).reshape(6,5), 
                    index=['Cat', 'Dog', 'Lion', 'Cow', 'Ant', 'Mouse'],
                    columns=['A', 'B', 'C', 'D', 'E'])
animals
animals.loc[['Cat'],['A']]             = np.nan
animals.loc[['Mouse'],['D']]           = np.nan
animals.loc[['Ant', 'Cow'],['B', 'C']] = np.nan
animals

