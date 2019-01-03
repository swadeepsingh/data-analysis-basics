
# coding: utf-8

# # Analyze IRIS Dateset 
# 
# 
# 
# ## About Dataset
# 
# 
# The Iris flower data is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis . It is sometimes called Anderson’s Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. Two of the three species were collected in the Gaspé Peninsula “all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus”.
# 
# 
# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimetres. 
# 

# ## Load Dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# load iris dataset
iris_df = pd.read_csv('iris.csv')
iris_df.head()


# In[3]:


iris_df.tail()


# In[4]:


iris_df.sample(5)


# In[5]:


# dataframe info 
iris_df.info()


# In[6]:


# dataframe stats
iris_df.describe()


# In[7]:


# dataframe shape
iris_df.shape


# In[8]:


# dataframe data types
iris_df.dtypes


# In[9]:


# no of unique values
iris_df.nunique()


# ## Clean Dataset

# In[10]:


# check for duplicates
dup_in_iris_df = iris_df.duplicated()
dup_in_iris_df.value_counts()


# In[11]:


# remove duplicates
iris_df.drop_duplicates(inplace=True)


# In[12]:


# re-check
iris_df.shape


# ## Statistical correlation of features

# In[13]:


# correlation map to show relationships between features\n",
f,ax = plt.subplots(figsize=(10, 7))
sns.heatmap(iris_df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.title("Relationships between features")
plt.show()


# ## Visualize dataset

# ### Boxplot which is going to be in the univariate form for each measurement.

# In[14]:


# box and whisker plots
iris_df.plot(kind='box', figsize=(15, 10), subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# ### Histogram representation of the univariate plots for each measurement

# In[15]:


# histograms
iris_df.hist(figsize=(15,10))
plt.show()


# ### Scatterplots of all pairs of attributes can be helpful to spot structured relationships between input variables.

# In[16]:


# scatter plot matrix
warnings.filterwarnings("ignore")
pd.scatter_matrix(iris_df, figsize=(15,10))
plt.show()


# ## Conclusion
# 
# ### Basic analysis of IRIS dataset using pandas
# 
# In this tutorial we have learn the following:
# 
# 1. load a dataset from csv file
# 2. various commands to explore dataset (info, describe, shape, head, tail, sample)
# 3. get information on unique data entries 
# 4. finding the duplicated in dataset
# 5. cleaning the dataset
# 6. correlation of features
# 7. visual representation of dataset
#     1. boxplot
#     2. histograms
#     3. scatter plots
