#!/usr/bin/env python
# coding: utf-8

# In[356]:


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


# ### Histogram matplotlib

# In[12]:


dataset1 = np.random.randn(100)
plt.hist(dataset1)


# In[17]:


dataset2 = np.random.randn(80)
plt.hist(dataset2,  color='indianred')


# In[20]:


plt.hist(dataset1, density=True, color='red', bins=20, alpha=0.5)
plt.hist(dataset2, density=True, color='green', bins=20, alpha=0.5)


# ### Seaborn

# In[21]:


data1 = np.random.randn(1000)
data2 = np.random.randn(1000)


# In[22]:


sns.jointplot(data1, data2)


# In[23]:


sns.jointplot(data1, data2, kind='hex')


# In[163]:


# Example sigmoid
def sigmoid(x):
     y = 1 / (1 + np.exp(-x))
     return y
x = np.linspace(-10, 15, 100)

y = sigmoid(x)
t = plt.plot(x, y,color = 'red',alpha=0.5)


# ### KDE (Kernal Density Estimation)

# In[38]:


# Manual method
dataset = np.random.randn(25)
np.sort(dataset)

# Size of dataset
np.size(dataset)

# Scale stuff
plt.ylim(0, 1)

sns.rugplot(dataset)

#plt.hist(dataset, density=True, color='red', bins=int(np.size(dataset)/1), alpha=0.5)
t = plt.hist(dataset, density=True, color='red', alpha=0.5)


# In[79]:


# Increment by 0.1
V = np.arange(1, 3, 0.1)
V

# Useful function to create evenly spaced markers, given count
arr1 = np.linspace(1,3,20)
arr1

# Now a meshgrid!!!
x, y = np.meshgrid(arr1, arr1)


# In[129]:


# value of the pdf at point 49.9 for a given mean 50, sigma 2 is 0.19922195704738202
# The value of the pdf at point x for a given mu, sigma

# Note that this is not a probability (= area under the pdf) 
# but rather the value of the pdf at the point x you pass to pdf(x)
# stats.norm(mu,sigma).pdf(some_value x )
stats.norm(50,2).pdf(49.9)

# Get a list
stats.norm(50,2).pdf(np.linspace((50-4),(50+4),40))


# random testing for below
# Create a kernel for each point and append to list


data_point = 1.5  # This is the mean
x_min  = data_point - 1
x_max  = data_point + 1
x_axis = np.linspace(x_min,x_max,100)

# sigma/bw
bw = 0.3

k = stats.norm(data_point, bw).pdf(x_axis)
k =  (k /k.max()) * 2  # Just to adjust scale
t = plt.plot(x_axis, k,color = 'red',alpha=0.5)


    
#Scale for plotting
# kernel = kernel / kernel.max()
# kernel = kernel * .4
# t = plt.plot(x_axis,kernel,color = 'grey',alpha=0.5)


# In[133]:


# Create another rugplot
sns.rugplot(dataset);

# Set up the x-axis for the plot
x_min = dataset.min() - 2
x_max = dataset.max() + 2

# 100 equally spaced points from x_min to x_max
x_axis = np.linspace(x_min,x_max,100)

# Set up the bandwidth, for info on this:
url = 'http://en.wikipedia.org/wiki/Kernel_density_estimation#Practical_estimation_of_the_bandwidth'

bandwidth = ((4*dataset.std()**5)/(3*len(dataset)))**.2


# Create an empty kernel list
kernel_list = []

# Plot each basis function
for data_point in dataset:
    
    # Create a kernel for each point and append to list
    kernel = stats.norm(data_point,bandwidth).pdf(x_axis)
    kernel_list.append(kernel)
    
    #Scale for plotting
    kernel = kernel / kernel.max()
    kernel = kernel * .4
    t = plt.plot(x_axis,kernel,color = 'grey',alpha=0.5)

t = plt.ylim(0,1)


# In[139]:


# To get the kde plot we can sum these basis functions.
# Plot the sum of the basis function
sum_of_kde = np.sum(kernel_list, axis=0)

# Plot figure
fig = plt.plot(x_axis, sum_of_kde, color='gray')

# Add the initial rugplot
t = sns.rugplot(dataset, color='indianred')

# Get rid of y-tick marks
#plt.yticks([])

# Set title
t = plt.suptitle("Sum of the Basis Functions")


# In[141]:


# Exact same stuff above 
sns.kdeplot(dataset)


# In[148]:


# We can adjust the bandwidth of the sns kde to make the kde plot more or less sensitive to high frequency

# Rugplot
t = sns.rugplot(dataset,color='black')

# Plot various bandwidths
for bw in np.arange(0.5,2,0.25):
    t = sns.kdeplot(dataset,bw=bw,label=bw)


# In[150]:


# We can also choose different kernels

kernel_options = ["biw", "cos", "epa", "gau", "tri", "triw"]

# More info on types
url = 'http://en.wikipedia.org/wiki/Kernel_(statistics)'

# Use label to set legend
for kern in kernel_options:
    t = sns.kdeplot(dataset,kernel=kern,label=kern)


# In[151]:


# We can also shade if desired
for kern in kernel_options:
    y = sns.kdeplot(dataset,kernel=kern,label=kern,shade=True,alpha=0.5)


# In[153]:


# For vertical axis, use the vertical keyword
y = sns.kdeplot(dataset,vertical=True)


# In[155]:


# Finally we can also use kde plot to create a cumulative distribution function (CDF) of the data

# URL for info on CDF
url = 'http://en.wikipedia.org/wiki/Cumulative_distribution_function'

t = sns.kdeplot(dataset,cumulative=True)


# In[280]:


# Let's create a new dataset - 

# Mean center of data
mean = [0,0]

# Diagonal covariance
cov = [[1,0],[0,100]]

# Create dataset using numpy
dataset2 = np.random.multivariate_normal(mean, cov, 1000)

# np.cov(dataset2.T) = The output of this will match matrix 'cov'


# Bring back our old friend pandas
dframe = pd.DataFrame(dataset2,columns=['X','Y'])

# Plot our dataframe
t = sns.kdeplot(dframe, shade=True)


# In[279]:


# Another way

t = sns.kdeplot(dframe['X'], dframe['Y'])


# In[282]:


# Change band width
t = sns.kdeplot(dframe, bw=1)


# In[284]:


# Silverman's (1986) rule of thumb
t = sns.kdeplot(dframe, bw='silverman')


# In[285]:


# We can also create a kde joint plot, simliar to the hexbin plots we saw before

sns.jointplot('X','Y',dframe,kind='kde')


# ### using distplot - combine different plots

# In[289]:


# Create datset
dataset = np.random.randn(500)

# Use distplot for combining plots, by default a kde over a histogram is shown
sns.distplot(dataset,bins=25)


# In[290]:


# hist, rug, and kde are all input arguments to turn those plots on or off
sns.distplot(dataset,rug=True,hist=False)


# In[291]:


# TO control specific plots in distplot, use [plot]_kws argument with dictionaries.

#Here's an example

sns.distplot(dataset,bins=25,
             kde_kws={'color':'indianred','label':'KDE PLOT'},
             hist_kws={'color':'blue','label':"HISTOGRAM"})


# In[292]:


# Create Series form dataset
ser1 = Series(dataset,name='My_DATA')

# Plot Series
sns.distplot(ser1,bins=25)


# ### box and violin plots

# In[102]:


# Create data

dframe = DataFrame(np.random.randn(250).reshape(50,5),  
                   columns=['Mon', 'Tue', 'Wed', 'Thu', 'Fri' ])

workers = [np.random.choice(['bob', 'nick', 'tom', 'harry']) for i in range(1,50)]
dframe['Name']  = Series(workers) 
dframe.columns.names = ['Days']
dframe.index.names = ['Readings']
dframe.head()

sns.boxplot( x=dframe["Name"], y=dframe["Mon"] )


# In[103]:


link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = sns.load_dataset('iris')
df.head()
sns.boxplot( x=df["species"], y=df["sepal_length"] )


# In[111]:


# Let's create two distributions
data1 = np.random.randn(10)
data2 = np.random.randn(10) + 10

# Now we can create a box plot
t = sns.boxplot([data1, data2], orient='v')


# In[126]:


df = pd.DataFrame({'a' :['one','one','two','two','one','two','one','one','one','two'], 
                   'b': [1,2,1,2,1,2,1,2,1,1], 
                   'c': [1,2,3,4,6,1,2,3,4,6]})
df

sns.boxplot(  y="b", x= "a", data=df,  orient='v' )

sns.boxplot(  y="b", data=df,  orient='v' )

# We create the figure with the subplots:
# Where axes is an array with each subplot.
# Then we tell each plot in which subplot we want them with the argument ax.

f, axes = plt.subplots(1, 2)
sns.boxplot(  y="b", x= "a", data=df,  orient='v' , ax=axes[0])
sns.boxplot(  y="c", x= "a", data=df,  orient='v' , ax=axes[1])


# In[146]:


# Let's create an example where a box plot doesn't give the whole picture

# Normal Distribution
data1 = stats.norm(0,5).rvs(100)

# Two gamma distributions concatenated together (Second one is inverted)
data2 = np.concatenate([stats.gamma(5).rvs(50)-1,
                        -1*stats.gamma(5).rvs(50)])

# Single data
sns.boxplot(data=[data1, data2],  orient='v' )


# In[158]:


# From the above plots, you may think that the distributions are fairly similar
# But lets check out what a violin plot reveals
sns.violinplot([data1, data2], orient='v')

# Much like a rug plot, we can also include the individual points, or sticks
f, axes = plt.subplots(1, 2)
sns.violinplot(data1,  orient='v' , ax=axes[0], inner="stick")
sns.violinplot(data2,  orient='v' , ax=axes[1], inner="stick")


# In[167]:


# Seaborn internal data set

tips = sns.load_dataset("tips")
tips.head()
sns.violinplot(x="day", y="total_bill", data=tips)


# ### Regression plots
# 

# In[176]:


# Seaborn comes with an example dataset to use as a pandas DataFrame
tips = sns.load_dataset("tips")
tips.head()

# Simple scatter plot with linear regression,
# which is an estimateed linear fit model to the data
t = sns.lmplot("total_bill", "tip", tips)


# In[183]:


# WE can also specify teh confidence interval to use for the linear fit

sns.lmplot("total_bill","tip",tips, ci=68) # 68% ci 


# In[188]:


# Just like before, we can use dictionaries to edit individual parts of the plot

sns.lmplot("total_bill", "tip", tips,
           scatter_kws={"marker": "o", "color": "indianred"},
           line_kws={"linewidth": 1, "color": "blue"});


# In[191]:


# WE can also check out higher-order trends
sns.lmplot("total_bill", "tip", tips,order=5,
           scatter_kws={"marker": "o", "color": "indianred"},
           line_kws={"linewidth": 1, "color": "blue"})


# In[192]:


# We can also not fit a regression if desired
sns.lmplot("total_bill", "tip", tips,fit_reg=False)


# In[215]:


# lmplot() also works on discrete variables, such as the percentage of the tip
tips = sns.load_dataset("tips")
# Create a new column for tip percentage
tips["tip_pect"]=100*(tips['tip']/tips['total_bill'])


#plot
sns.lmplot("size", "tip_pect", tips);

tips.head().style.format({'tip_pect': "%{:.2f}"})



# In[216]:


# We can also add jitter to this

#Info link
url = "http://en.wikipedia.org/wiki/Jitter"

#plot
sns.lmplot("size", "tip_pect", tips,x_jitter=.1);


# In[219]:


# We can also estimate the tendency of each bin (size of party in this case)
sns.lmplot("size", "tip_pect", tips, x_estimator=np.mean);

# x_estimator => Apply this function to each unique value of x and plot the resulting estimate. This is useful when x is a discrete variable. 
# looks like there is more variance for party sizes of 1 then 2-4


# In[222]:


# We can use the hue facet to automatically define subsets along a column
tips.head()

# Plot, note the markers argument
sns.lmplot("total_bill", "tip_pect", tips, hue="sex",markers=["x","*"])


# In[223]:


# Does day make a difference?
sns.lmplot("total_bill", "tip_pect", tips, hue="day")


# In[224]:


# Finally it should be noted that Seabron supports LOESS model fitting
url = 'http://en.wikipedia.org/wiki/Local_regression'

sns.lmplot("total_bill", "tip_pect", tips, lowess=True, line_kws={"color": 'black'});


# In[225]:


# The lmplot() we've been using is actually using a lower-level function, regplot()

sns.regplot("total_bill","tip_pect",tips)


# In[231]:


# reg_plot can be added to existing axes without modifying anything in the figure

# Create figure with 2 subplots
fig, (axis1,axis2) = plt.subplots(1,2,sharey =True)

sns.regplot("total_bill","tip_pect",tips,ax=axis1)

sns.violinplot(x='size', y='tip_pect',data=tips,ax=axis2)


# ### Heat maps

# In[267]:


# Pandas data frame example

df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                   "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                   "C": ["small", "large", "large", "small","small", "large", "small", "small", "large"],
                   "D": [1, 2, 2, 3, 3, 4, 5, 6, 7]})
df

table = pd.pivot_table(df, values='D', index=['B'], columns=['C'], aggfunc=np.sum)

table


# In[290]:


# Again seaborn comes with a great dataset to play and learn with
flight_dframe = sns.load_dataset('flights')
flight_dframe.head()

flight_pivot = pd.pivot_table(flight_dframe, values='passengers', index=['month'], columns=['year'], aggfunc=np.sum)
flight_pivot.head()


# In[274]:


# This dataset is now in a clear format to be dispalyed as a heatmap
sns.heatmap(flight_pivot)


# In[352]:


# seaborn will automatically try to pick the best color scheme for your dataset, whether is be diverging or converging colormap

sns.heatmap(flight_pivot,annot=True, fmt='d')


# ### Heat map example

# In[353]:


data = np.random.rand(10,10)*5
ax = sns.heatmap(data, vmin=0,vmax=5,center=2.5,cmap="RdBu_r")


# In[350]:



# We can choose our own 'center' for our colormap
flight_dframe.head()

flight_dframe.loc[:,'passengers'].max()


sns.heatmap(flight_pivot, 
            center=flight_dframe.loc[:,'passengers'].mean(), 
            vmin=flight_dframe.loc[:,'passengers'].min(), 
            vmax=flight_dframe.loc[:,'passengers'].max(),
            cmap="RdBu_r")


# In[355]:


# Pandas cells manipulation
url= 'https://pythonhow.com/accessing-dataframe-columns-rows-and-cells'


# In[386]:


# heatmap() can be used on an axes for a subplot to create more informative figures

flight_dframe = sns.load_dataset('flights')
flight_dframe.head()

flight_pivot = pd.pivot_table(flight_dframe, values='passengers', index=['month'], columns=['year'], aggfunc=np.sum)
flight_pivot

df1 = pd.DataFrame(flight_pivot.sum())


df1.reset_index(inplace = True) 
df1 = df1.rename(columns={0: 'Flights'})
df1

f, (axis1,axis2) = plt.subplots(2,1)

sns.barplot(x='year', y='Flights', data=df1, ax=axis1)
sns.heatmap(flight_pivot, cmap='Blues', cbar_kws={"orientation": "horizontal"}, ax=axis2)


# In[390]:


# Finally we'll learn about using a clustermap

# Clustermap will reformat the heatmap so similar rows are next to each other
sns.clustermap(flight_pivot)


# In[391]:


# Let's uncluster the columns
sns.clustermap(flight_pivot,col_cluster=False)


# In[392]:


# standard_scale : Either 0 (rows) or 1 (columns). Whether or not to standardize that dimension, meaning for each row or column, subtract the minimum and divide each by its maximum
sns.clustermap(flight_pivot, standard_scale=0)


# In[393]:


# Finally we can also normalize the rows by their Z-score.
# This subtracts the mean and devides by the STD of each column, then teh rows have amean of 0 and a variance of 1
sns.clustermap(flight_pivot,z_score=1)


# In[ ]:




