#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# In[2]:


dataset = pd.read_csv(r"diabetes (1).csv")


# In[3]:


dataset.head()


# In[4]:


dataset.shape


# In[5]:


dataset.describe()


# In[6]:


sns.countplot(x = 'Outcome', data = dataset)


# In[7]:


corr_mat = dataset.corr()
sns.heatmap(corr_mat, annot = True)
plt.show()


# In[8]:


dataset.isna().sum()


# In[9]:


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# In[10]:


X[0]


# In[11]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[12]:


x_train.shape


# In[13]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[15]:


x_train[0]


# In[16]:


#support vector machine
from sklearn.svm import SVC
SVC_model=SVC()
SVC_model.fit(x_train,y_train)


# In[18]:


SVC_pred=SVC_model.predict(x_test)


# In[19]:


from sklearn import metrics
print("Accuracy score= ",format(metrics.accuracy_score(y_test,SVC_pred)))


# In[25]:


#random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(x_train,y_train)


# In[26]:


rfc_train=rfc.predict(x_train)
from sklearn import metrics

print("Accuracy score= ", format(metrics.accuracy_score(y_train,rfc_train)))


# In[27]:


from sklearn import metrics

predictions=rfc.predict(x_test)
print("Accuracy Score= ", format(metrics.accuracy_score(y_test,predictions)))


# In[30]:


#knn
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=25, metric="minkowski")
knn.fit(x_train,y_train)


# In[31]:


y_pred=knn.predict(x_test)


# In[32]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


# In[33]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[34]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(x_train,y_train)


# In[40]:


from sklearn import metrics

predictions = dtree.predict(x_test)
print("Accuracy_score= ", format(metrics.accuracy_score(y_test,predictions)))


# In[37]:


#gradient booster
from sklearn.ensemble import GradientBoostingClassifier
xgb_model = GradientBoostingClassifier(n_estimators = 300, max_features = 2, max_depth = 9,random_state = 0)
xgb_model.fit(x_train,y_train)


# In[38]:


from sklearn import metrics

xgb_pred=xgb_model.predict(x_test)
print("Accuracy_score= ", format(metrics.accuracy_score(y_test,xgb_pred)))


# In[ ]:




