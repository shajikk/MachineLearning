#!/usr/bin/env python
# coding: utf-8

# ## Solve equations

# In[33]:


# Print all the interactive output without resorting to print, not only the last result
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[21]:


from sympy.interactive import printing
printing.init_printing(use_latex=True)
from sympy import Eq, solve_linear_system, Matrix
from numpy import linalg
import numpy as np
import sympy as sp

print ("Using Sympy:")

# Set up display part
eq1 = sp.Function('eq1')
eq2 = sp.Function('eq2')
x,y  = sp.symbols('x y')

eq1 = Eq(2*x - y, -4)
eq2 = Eq(3*x - 1, -2)
display(eq1)
display(eq2)

row1 = [2, -1, -4]
row2 = [3, -1, -2]

system = Matrix((row1, row2))
display(system)

display(solve_linear_system(system, x, y))


# In[27]:


print ("Using Numpy:")

nrow1 = [2, -1]
nrow2 = [3, -1]

nmat = np.array([nrow1, nrow2])
cons = np.array([-4, -2])

answer = linalg.solve(nmat, cons)
display(answer)

print("x = %s, y = %s" %(int(answer[0]), int(answer[1])))


# In[34]:


nrow1 = [2, -1, 3, 4, 5]
nrow2 = [3, -1, 8, 9, 10]
nrow3 = [12, -1, 3, 4, 5]
nrow4 = [3, -11, 8, 9, 1]
nrow5 = [9, -1, 8, 7, 1]

nmat = np.array([nrow1, nrow2, nrow3, nrow4, nrow5 ])
nmat
cons = np.array([-4, -2, 7, 8, 9])

answer = linalg.solve(nmat, cons)
display(answer)


# In[ ]:




