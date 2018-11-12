#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Print all the interactive output without resorting to print, not only the last result
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# Stats
from scipy import stats

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Coverance experiments

# In[2]:


# Array with  X1, X2, X3, X4 features.
X = np.array([ [0.1, 0.3, 0.4, 0.8],
               [3.2, 2.4, 2.4, 0.1],
               [10., 8.2, 4.3, 2.6],
             ])

pd.DataFrame(X,columns=['X1','X2', 'X3', 'X4'])

#np.cov(X.T)
# Covarience matrix
pd.DataFrame(np.cov(X.T), columns=['X1','X2', 'X3', 'X4'], index=['X1','X2', 'X3', 'X4'])

# Covarience matrix between X1 and X2
pd.DataFrame(np.cov(X.T[0], X.T[1]),columns=['X1','X2'], index=['X1','X2'])


# In[3]:


mean = [0,0]

#  covariance Matrix
cov = [[8, 0],
       [0, 9]]


dataset = np.random.multivariate_normal(mean, cov, 100)


t = plt.scatter(dataset.T[0], dataset.T[1])


# In[4]:


# Cook up some data

mean = [0,0]

a = np.arange(50)
a = a * 0.08 + a
a = a - np.max(a)/2
b = a * 2.5 + (a**2) * 0.01 - (a**3) * 0.001 + 8
b = b - np.max(b)/2

t = plt.scatter(a, b)

pd.DataFrame(np.cov(a, b),columns=['X1','X2'], index=['X1','X2'])

# feed the covariance matrix into random number generator 


cov = np.cov(a, b)
dataset = np.random.multivariate_normal([0, 0], cov, 50) 
t = plt.scatter(dataset.T[0], dataset.T[1])

#sns.kdeplot(dframe)


# In[27]:


def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color

def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

def highlight_y():
    '''
    highlight yellow.
    '''
    return ['background-color: yellow']


# Make 'rs' and then use 'rs'. instead of np.random.
# pass 'rs' around, and have multiple random streams

rs = np.random.RandomState(0)

df = pd.DataFrame(rs.randn(10, 10), columns=list('ABCDEFGHIJ'))

corr = df.corr()

corr.style.applymap(color_negative_red).apply(highlight_max)

#corr.style

corr.style.background_gradient().set_precision(2)

corr.style.format({'B': "{:+.2f}", 'D': '{:+.2f}'})

df.style.format({"A": lambda x: "±{:.2f}".format(abs(x))})

