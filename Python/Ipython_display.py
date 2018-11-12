#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import IPython

framerate = 44100
t = np.linspace(0,5,framerate*5)
# data = np.sin(2*np.pi*220*t) + np.sin(2*np.pi*224*t)
data = np.sin(2*np.pi*500*t)
IPython.display.Audio(data,rate=framerate)


# In[8]:


from IPython.display import display, Math, Latex
display(Math(r'F(k) = \int_{-\infty}^{\infty} f(x) e^{2\pi i k} dx'))


# In[ ]:




