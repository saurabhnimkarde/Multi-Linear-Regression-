#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from statsmodels.graphics.regressionplots import influence_plot
import matplotlib.pyplot as plt
toyota = pd.read_csv("ToyotaCorolla.csv")
toyota.shape


# In[2]:


toyota1= toyota.iloc[:,[2,3,6,8,12,13,15,16,17]]


# In[3]:


toyota1.rename(columns={"Age_08_04":"Age"},inplace=True)


# In[4]:


toyota1.corr()


# In[5]:


import seaborn as sns
sns.set_style(style ="darkgrid")
sns.pairplot(toyota1)


# In[7]:


import statsmodels.formula.api as smf
model1 = smf.ols('Price~Age+KM+HP+Doors+cc+Gears+Quarterly_Tax+Weight', data=toyota1).fit()


# In[8]:


model1.summary()


# In[9]:


model_influence = model1.get_influence()
(c, _) = model_influence.cooks_distance
c


# In[10]:


fig = plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(toyota)), np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()


# In[11]:


np.argmax(c), np.max(c)


# In[12]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model1)
plt.show()


# In[13]:


k = toyota1.shape[1]
n = toyota1.shape[0]
leverage_cutoff = 3*((k+1)/n)
leverage_cutoff


# In[14]:


toyota_new = toyota1.drop(toyota1.index[[80,960,221,601]],axis=0).reset_index()
toyota3=toyota_new.drop(['index'], axis=1)
toyota3


# In[15]:


import statsmodels.formula.api as smf
model2 = smf.ols('Price~Age+KM+HP+Doors+cc+Gears+Quarterly_Tax+Weight', data=toyota3).fit()


# In[16]:


model2.summary()


# In[17]:


finalmodel = smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data = toyota3).fit()
finalmodel.summary()


# In[18]:


finalmodel_pred = finalmodel.predict(toyota3)


# In[19]:


plt.scatter(toyota3["Price"],finalmodel_pred,color='blue');plt.xlabel("Observed values");plt.ylabel("Predicted values")


# In[20]:


plt.scatter(finalmodel_pred, finalmodel.resid_pearson,color='red');
plt.axhline(y=0,color='blue');
plt.xlabel("Fitted values");
plt.ylabel("Residuals")


# In[21]:


plt.hist(finalmodel.resid_pearson)


# In[22]:


import pylab
import scipy.stats as st
st.probplot(finalmodel.resid_pearson, dist='norm',plot=pylab)


# In[23]:


new_data=pd.DataFrame({'Age':25,'KM':40000,'HP':80,'cc':1500,'Doors':3,'Gears':5,'Quarterly_Tax':180,'Weight':1050}, index=[1])


# In[24]:


new_data


# In[25]:


finalmodel.predict(new_data)


# In[26]:


pred_y=finalmodel.predict(toyota1)


# In[27]:


pred_y


# # training and testing the data 

# In[28]:


from sklearn.model_selection import train_test_split


# In[30]:


train_data,test_Data= train_test_split(toyota1,test_size=0.3)

finalmodel1 = smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data = train_data).fit()
finalmodel1.summary()


# In[31]:


finalmodel_pred = finalmodel1.predict(train_data)


# In[32]:


finalmodel_res = train_data["Price"]-finalmodel_pred


# In[33]:


finalmodel_rmse = np.sqrt(np.mean(finalmodel_res*finalmodel_res))


# In[34]:


finalmodel_testpred = finalmodel1.predict(test_Data)


# In[35]:


finalmodel_testres= test_Data["Price"]-finalmodel_testpred


# In[36]:


finalmodel_testrmse = np.sqrt(np.mean(finalmodel_testres*finalmodel_testres))


# In[37]:


finalmodel_testrmse


# In[ ]:




