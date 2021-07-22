#!/usr/bin/env python
# coding: utf-8

# # Lecture 4: Introduction to Regression 
# ### Data Science 1: CS 109A/STAT 121A/AC 209A/ E 109A <br> Instructors: Pavlos Protopapas, Kevin Rader, Rahul Dave, Margo Levine
# #### Harvard University <br> Fall 2017 <br> 
# 
# ---
# 

# In[2]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import ElasticNet, LassoLars, Ridge, LinearRegression, Lasso
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
 

sns.set(style="ticks")
#get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 100)

sns.set_context('poster')
# import matplotlib.pylab as pylab
# params = {'legend.fontsize': 'large',
#           'figure.figsize': (15, 5),
#          'axes.labelsize': 'large',
#          'axes.titlesize':'large',
#          'xtick.labelsize':'large',
#          'ytick.labelsize':'large'}
# pylab.rcParams.update(params)}


# ## Step #1: Load and Explore Data

# In[3]:


# GET THE FULL DATA SET FROM 
 
df = pd.read_csv('spring.csv' ) # DO WE NEED THE LOW MEMORY 


 
 

# ## Step #2: Modeling the Data

# In[10]:


#X = df.drop(['station_id','pm2_5_reported'], axis = 1) # Features 
df = df.drop(['station_id' ], axis = 1) # Features 


df.head()


# In[50]:


corr=df.corr()
print(corr)
sns.heatmap(corr)

plt.savefig('correlation.png')


# In[12]:


fig = plt.figure(figsize = (20, 25))
j = 0
 

colors=['g','y','b','r','c','g']
for i in df.columns:
    plt.subplot(6, 4, j+1)
    j += 1
    sns.distplot(df[i],  color=colors[j-1],label = df[i].name)
     
    plt.legend(loc='best')
fig.suptitle('Distribution of variables')
fig.tight_layout()
fig.subplots_adjust(top=0.95)
#plt.show()
plt.savefig('features-distribution.png')


# In[13]:






X = df.drop(['pm2_5_reported'], axis = 1) # Features 


#Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)


Y=df.pm2_5_reported

# Dividing dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

#print(X_train.shape)
#print(X_test.shape)
 
 


# ---

# ## Step #3: Evaluate and Interpret the Model
 


#Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, Y_train)

 

pred_test_lr= lr.predict(X_test)
print('-------------- LR Results -------------------') 
print('MSE:  ', (mean_squared_error(Y_test,pred_test_lr))) 
print('RMSE:  ',np.sqrt((mean_squared_error(Y_test,pred_test_lr))) )
 
print('R2-score:  ',r2_score(Y_test, pred_test_lr))
 

print('Coefficients:  ',lr.coef_)
print('Intercept:  ',lr.intercept_)
  
plt.figure(figsize=(12,8))
plt.xlabel("Predicted value with Linear Regression",fontsize=20)
plt.ylabel("Actual y-values",fontsize=20)
plt.grid(1)
plt.scatter(pred_test_lr,Y_test,edgecolors=(0,0,0),lw=2,s=80)
plt.plot(pred_test_lr,pred_test_lr, 'k--', lw=2)
plt.savefig('LR-predictedvsActual.png')


# In[52]:


#Elastic Net Model
model_enet = ElasticNet(alpha = 0.05)
model_enet.fit(X_train, Y_train) 


#print(model_enet.coef_)
#print(model_enet.intercept_)
 

pred_test_enet= model_enet.predict(X_test)
print('-------------- ENET Results -------------------') 
print('MSE:  ', (mean_squared_error(Y_test,pred_test_enet))) 
print('RMSE:  ',np.sqrt((mean_squared_error(Y_test,pred_test_enet))) )
 
print('R2-score:  ',r2_score(Y_test, pred_test_enet))
print('Coefficients:  ',model_enet.coef_)
print('Intercept:  ',model_enet.intercept_)


plt.figure(figsize=(12,8))
plt.xlabel("Predicted value with Elastic Net",fontsize=20)
plt.ylabel("Actual y-values",fontsize=20)
plt.grid(1)
plt.scatter(pred_test_enet,Y_test,edgecolors=(0,0,0),lw=2,s=80)
plt.plot(pred_test_enet,pred_test_enet, 'k--', lw=2)
plt.savefig('ENetR-predictedvsActual.png')


# In[17]:


regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, Y_train)
#regr.predict(X_test[:2])
#regr.score(X_test, y_test)


# In[18]:





# In[22]:



pred_test_mlp= regr.predict(X_test)


# In[53]:


print('-------------- MLP Results -------------------') 
print('MSE:  ', (mean_squared_error(Y_test,pred_test_mlp))) 
print('RMSE:  ',np.sqrt((mean_squared_error(Y_test,pred_test_mlp))) )
 
print('R2-score:  ',r2_score(Y_test, pred_test_mlp))
print('Coefficients:  ',regr.coefs_)
print('Intercept:  ',regr.intercepts_)


plt.figure(figsize=(12,8))
plt.xlabel("Predicted value with MLP",fontsize=20)
plt.ylabel("Actual y-values",fontsize=20)
plt.grid(1)
plt.scatter(pred_test_mlp,Y_test,edgecolors=(0,0,0),lw=2,s=80)
plt.plot(pred_test_mlp,pred_test_mlp, 'k--', lw=2)
plt.savefig('MLPR-predictedvsActual.png')


# In[25]:





# In[54]:


#regr = make_pipeline(StandardScaler(), SVR(C=256, epsilon=0.2))

regr = SVR(kernel='linear',C=256, epsilon=0.2)
 
regrV=regr.fit(X_train, Y_train)
 
pred_test_SVR= regrV.predict(X_test)

print('-------------- SVR Results -------------------') 
print('MSE:  ', (mean_squared_error(Y_test,pred_test_SVR))) 
print('RMSE:  ',np.sqrt((mean_squared_error(Y_test,pred_test_SVR))) )
 
print('R2-score:  ',r2_score(Y_test, pred_test_SVR))
print('Coefficients:  ',regr.coef_)
print('Intercept:  ',regr.intercept_)


plt.figure(figsize=(12,8))
plt.xlabel("Predicted value with SVR",fontsize=20)
plt.ylabel("Actual y-values",fontsize=20)
plt.grid(1)
plt.scatter(pred_test_SVR,Y_test,edgecolors=(0,0,0),lw=2,s=80)
plt.plot(pred_test_SVR,pred_test_SVR, 'k--', lw=2)
plt.savefig('SVR-predictedvsActual.png')

