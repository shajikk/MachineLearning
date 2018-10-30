#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

my_list1 = [ 1 , 3, 5, 7, 9]


# In[4]:


np.array(my_list1)


# In[5]:


recall last


# In[6]:


recall 4


# In[ ]:


np.array(my_list1)


# In[9]:


a = np.array([2, 3, 5])


# In[10]:


a = np.array([[2, 3, 5], [3, 4, 5]])


# In[11]:


a


# In[14]:


a.shape


# In[15]:


a.dtype


# In[17]:


np.zeros(3)


# In[19]:


np.zeros(3).dtype


# In[20]:


np.ones([5,4])


# In[21]:


np.eye(5)


# In[24]:


np.eye(5)


# In[27]:


np.arange(1,2,.1)


# In[28]:


5/2


# In[30]:


a = np.arange(2,9,1)


# In[31]:


a*a


# In[32]:


1/a


# In[33]:


a ^ 3


# In[34]:


a ** 3


# In[35]:


a


# In[36]:


a[0]


# In[37]:


a.size


# In[39]:


a[2:7]


# In[40]:


b = np.arange(1, 10, 1)


# In[41]:


b[:]


# In[42]:


b[3:]


# In[43]:


b[:3]


# In[55]:


# Very important
p = np.arange(1, 20, 2)
s = p[1:3]
s[:] = 0
p


# In[57]:


p = np.arange(1, 20, 2)
s = p[1:3]
s[:] = [100, 101]
p


# In[59]:


p1 = p.copy()
p1


# In[64]:


m1 = np.array([[22, 21, 20], [18, 20 , 3], [32, 1, 0]])
m1[1]
m1[1][0]


# In[65]:


m1


# In[102]:


# Reshape array - by slicing parts of an exsisting array
m1[:2,1:]


# In[76]:


# Row slice (select Row)
m1[2]
# Column slice (select Column)
m1[:,1]


# In[84]:


ar = np.zeros((10,10))
for i in range (ar.shape[1]):
    ar[i] = i


# In[88]:


ar[5] = [1, 2, 3, 4, 5, 6, 7, 8, 9 , 10 ]
ar


# In[101]:


# Fancy indexing. Return a matrix
# Note that rows are not in order
ar[[2, 8, 6, 4]]


# In[109]:


# Create a new matrix
a = np.arange(2, 100, 3).reshape(11, 3)
a


# In[112]:


# Transpose a matrix
a.T


# In[114]:


np.dot(a, a.T)


# In[128]:


# Create 6x3 matrix using reshape.
a = np.arange(2, 38, 1).reshape(2,3,6)
a


# In[120]:


a.size


# In[139]:


#Random matrix
np.floor(np.random.rand(2, 2, 2)*72)


# In[143]:


a= np.floor(np.random.rand(10, 5)*72)
b = a.T


# In[144]:


#Universal functions
np.sqrt(np.floor(np.random.rand(5, 5)*72))


# In[145]:


np.sqrt(np.floor(np.random.rand(3, 3)*10))


# In[149]:


# Normal distribution of random numbers
A = np.random.randn(10)
B = np.random.randn(10)
np.add(A, B)
np.maximum(A, B)


# In[151]:


import matplotlib.pyplot as plt


# In[167]:


t = np.arange(-5, 5, 0.025)
s = np.sin(t)
plt.plot(t,s)
plt.xlabel('x values')
plt.ylabel('transformed values')
plt.title('Simple stuff')
plt.grid(True)


# In[177]:


x = np.arange(-5, 5, 0.5)
y = np.arange(-5, 5, 0.5)
plt.plot(x,y) # Simple line
plt.plot(x,y, marker='.', color='r', linestyle='none')


# In[346]:


# Another form of meshgrid
V = np.arange(1, 6, 1)
x = np.array([V]*5)
y = np.array([V]*5).T
plt.plot(x,y, marker='.', color='k', linestyle='none')
V = np.arange(1, 6, 1)
x, y = np.meshgrid(V, V)
plt.plot(x,y, marker='.', color='k', linestyle='none')


# In[348]:


# Mesh grid usage (3d)
V = np.arange(1, 6, 0.1)
x, y = np.meshgrid(V, V)
#plt.plot(x,y, marker='.', color='k', linestyle='none')
r = np.sin(np.sqrt(x**2+y**2))


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection='3d')
#surf = ax.plot_surface(x, y, r, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#surf = ax.plot_surface(x, y, r)
surf = ax.plot_surface(x, y, r, cmap=cm.coolwarm)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


# In[349]:


# Mesh grid usage imshow  # Some kind of RGB ?
V = np.arange(-6, 6, 0.1)
x, y = np.meshgrid(V, V)
z = np.sin(x) + np.sin (y)
plt.imshow(z)


# In[351]:


# Where example application

x1 = np.floor(np.random.randn(5, 4)*10)
y1 = np.floor(np.random.randn(5, 4)*10)

np.where(x1>y1, 0, 1)


# In[352]:


# Another : Multi diam example
x = np.floor(np.random.randn(5, 4)*10)
y = np.floor(np.random.randn(5, 4)*10)
cond = np.less(x,y)

x1 = np.floor(np.random.randn(5, 4)*10)
y1 = np.floor(np.random.randn(5, 4)*10)

np.where(cond, x1, y1)


# In[360]:


#Sum of a matrix
x = np.floor(np.random.randn(5, 4)*10)
x.sum()
x.sum(axis=0) # Rowise sum
x.sum(axis=1) # Columnwise sum
x.mean() # Mean 
x.var() # Variance


# In[355]:


# Search for Any / All True
x = np.array([True, False, True])
x.any()
x.all()


# In[358]:


#Sort
a = np.floor(np.random.randn(1, 25)*10)
a.sort() # Sort a
a
np.unique(a) # Uniquify


# In[1]:


get_ipython().system('pwd')


# In[2]:


get_ipython().system('pwd')


# In[ ]:




