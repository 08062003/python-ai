#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
print('Python: {}'.format(sys.version))
import scipy
print('Scipy: {}'.format(scipy.__version__))
import numpy
print('Numpy: {}'.format(numpy.__version__))
import matplotlib
print('Matplotlib: {}'.format(matplotlib.__version__))
import pandas
print('Pandas: {}'.format(pandas.__version__))
import sklearn
print('Sklearn: {}'.format(sklearn.__version__))


# In[4]:


import pandas
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier


# In[5]:


#loading the data
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=read_csv(url,names=names)


# In[6]:


#dimensions of the dataset
print(dataset.shape)


# In[7]:


#take a peek at the data
print(dataset.head(20))


# In[8]:


#statistical summary
print(dataset.describe())


# In[9]:


#class distribution
print(dataset.groupby('class').size())


# In[10]:


#unvariate plots-box and whisker plots
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
pyplot.show()


# In[11]:


#histogram of the variable
dataset.hist()
pyplot.show()


# In[12]:


#multivariate plots
scatter_matrix(dataset)
pyplot.show()


# In[23]:


#creating a validation dataset
#splitting dataset
array=dataset.values
x=array[:,0:4]
y=array[:,4]
x_train,x_validation,y_train,y_validation=train_test_split(x,y,test_size=0.2,random_state=42,shuffle=True)


# In[24]:


#Logistic Regression
#Linear Discriminant Analysis
#K-Nearest neighbors
#Classification and Regression Trees
#Gaussian Naive Bayes
#Support Vector Machines
#building models
models=[]
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))


# In[1]:


#evaluate the created models
results=[]
names=[]
for name,model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=42,shuffle=True)
    cv_results=cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)'%(name,cv_results.mean(),cv_results.std()))



# In[28]:


#compare our models
pyplot.boxplot(results,labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()


# In[26]:


#make predictions on svm
model=SVC(gamma='auto')
model.fit(x_train,y_train)
predictions=model.predict(x_validation)


# In[29]:


#evaluate our predictions
print(accuracy_score(y_validation,predictions))
print(confusion_matrix(y_validation,predictions))
print(classification_report(y_validation,predictions))


# In[ ]:




