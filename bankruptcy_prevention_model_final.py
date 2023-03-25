#!/usr/bin/env python
# coding: utf-8

# #**Bankruptcy Prevention System**

# Problem Statement
#   
# Business Objective:
# This is a classification project, since the variable to predict is binary (bankruptcy or non-bankruptcy). The goal here is to model the probability that a business goes bankrupt from different features.
# 
# 
#   
# 

# Importing libraries

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


# In[2]:


df = pd.read_csv("bankruptcy-prevention.csv", sep =';')
#loading data and use seperator for seperate the columns


# In[3]:


df


# The data file contains 7 features about 250 companies
# 
# The data set includes the following variables:
# industrial_risk: 0=low risk, 0.5=medium risk, 1=high risk.
# management_risk: 0=low risk, 0.5=medium risk, 1=high risk.
# financial flexibility: 0=low flexibility, 0.5=medium flexibility, 1=high flexibility.
# credibility: 0=low credibility, 0.5=medium credibility, 1=high credibility.
# competitiveness: 0=low competitiveness, 0.5=medium competitiveness, 1=high competitiveness.
# operating_risk: 0=low risk, 0.5=medium risk, 1=high risk.
# class: bankruptcy, non-bankruptcy (target variable).
# 

# In[4]:


df.columns


# In[5]:


df= df.rename({' management_risk': 'management risk',' financial_flexibility':'financial flexibility',' credibility':'credibility',' competitiveness':'competitiveness',' operating_risk':'operating risk',' class':'class'}, axis=1)
df.columns


# # Exploratory Data Analysis

# In[6]:


df.head(5)


# In[7]:


df.tail(5)


# In[8]:


df.nunique()


# In[9]:


df.info()


# In[10]:


df.shape


# **Checking data types**

# In[11]:


df.dtypes


# Data Type Conversion

# In[12]:


df[df['class']=="bankruptcy"]


# In[13]:


df[df['class']=="non-bankruptcy"]


# In[14]:


df["class"] = df["class"].replace("bankruptcy",0)
df["class"] = df["class"].replace("non-bankruptcy",1)


# In[15]:


df


# In[16]:


df.value_counts('class') 


# In[17]:


df['class'].value_counts().plot.bar()


# In[18]:


df['class'].value_counts().plot(kind='pie',autopct='%.2f')
plt.show()


# In[19]:


df.dtypes


# Converted Class data type object to int

# **Null values**

# In[20]:


df.isnull()


# In[21]:


df.isnull().sum()


# In[22]:


sns.heatmap(df.isnull(), annot = True , cmap = 'rainbow_r')


# In given dataset, their is no any null values

# **Duplicate Values**

# In[23]:


df.duplicated()


# In[24]:


df[df.duplicated()]


# In[25]:


df[df.duplicated()].shape


# There are 147 duplicate values are present in dataset.
# 
# comapnies may have same values so we can't drop them

# #outlier Detection

# In[26]:


df.head(5)


# In[27]:


df.describe(percentiles=[0.10,0.20,0.90,0.99,0.995])


# In[28]:


df[	"industrial_risk"].describe(percentiles=[0.10,0.20,0.90,0.99,0.995])


# In[29]:


data_box=df
plt.boxplot(data_box.industrial_risk)


# #Visualization

# In[30]:


df.value_counts('management risk') 


# In[31]:


df['management risk'].value_counts().plot.bar()


# In[32]:


df['management risk'].value_counts().plot(kind='pie',autopct='%.2f')
plt.show()


# In[33]:


df.value_counts('financial flexibility') 


# In[34]:


df['financial flexibility'].value_counts().plot.bar()


# In[35]:


df['financial flexibility'].value_counts().plot(kind='pie',autopct='%.2f')
plt.show()


# In[36]:


df.value_counts('credibility') 


# In[37]:


df['credibility'].value_counts().plot.bar()


# In[38]:


df['credibility'].value_counts().plot(kind='pie',autopct='%.2f')
plt.show()


# In[39]:


df.value_counts('competitiveness') 


# In[40]:


df['competitiveness'].value_counts().plot.bar()


# In[41]:


df['competitiveness'].value_counts().plot(kind='pie',autopct='%.2f')
plt.show()


# In[42]:


df.value_counts('operating risk') 


# In[43]:


df['operating risk'].value_counts().plot.bar()


# In[44]:


df['operating risk'].value_counts().plot(kind='pie',autopct='%.2f')
plt.show()


# # Scatter plot and Correlation analysis

# scatter plot

# In[45]:


pd.plotting.scatter_matrix(df)
sns.pairplot(df)


# In[46]:


df.corr()


# In[47]:


#Correlation
Correlation=df.corr()


# In[48]:


sns.heatmap(Correlation, annot = True , cmap = 'rainbow_r')


# ### Model Building

# __Let's create multiple models one by one then we will cross_validate one by one to avoid Overfiting and Underfiting.
# Then will pick 2 algorithms with the best accuracy and improve accuracy of the particular algorithm using hyperparameter tuning__

# In[85]:


# Importing Necessary Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf


# In[50]:


df.head()


# In[51]:


# Converting dataframe into array
data = df.values
data


# In[52]:


# Selecting Target 
X=data[:,0:6]
Y=data[:,6]
print(X.shape,Y.shape)


# In[53]:


X


# In[54]:


Y


# In[55]:


# Data Split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.30,random_state=42)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# In[56]:


# Creating empty list for accuracy
acc = []
model = []


# ### 1.Logistic Regression

# In[57]:


lg = LogisticRegression()

lg.fit(x_train, y_train)

pred_lg = lg.predict(x_test)

print("Accuracy Score of Logistic Regression model is", accuracy_score(y_test, pred_lg)*100)

lg_scores = cross_val_score(lg, X, Y, cv = 10)# cross validating the model
print("\n")
print(lg_scores)
# accuracy scores of each cross validation cycle
print(f"\nMean of accuracy score for Logistic Regression is {lg_scores.mean()*100}\n")

x = metrics.accuracy_score(y_test, pred_lg)
acc.append(x)
model.append('lg')


# In[58]:


print(classification_report(y_test, pred_lg))


# In[59]:


confusion_matrix(y_test, pred_lg)


# ### 2.DecisionTreeClassifier

# In[60]:


dtc = DecisionTreeClassifier()

dtc.fit(x_train, y_train)

pred_dtc = dtc.predict(x_test)

print("Accuracy Score of Decision Tree Classifier model is", accuracy_score(y_test, pred_dtc)*100)

dtc_scores = cross_val_score(dtc, X, Y, cv = 10)
print("\n")
print(dtc_scores)
print(f"\nMean of accuracy score for Decision Tree Classifier is {dtc_scores.mean()*100}\n")

x = metrics.accuracy_score(y_test, pred_dtc)
acc.append(x)
model.append('dtc')


# In[61]:


print(classification_report(y_test, pred_dtc))


# In[62]:


confusion_matrix(y_test, pred_dtc)


# ### 3.KNN algorithm

# In[63]:


knc = KNeighborsClassifier(n_neighbors = 5)

knc.fit(x_train, y_train)

pred_knc = knc.predict(x_test)

print("Accuracy Score of K-Nearest Neighbour Classifier model is", accuracy_score(y_test, pred_knc)*100)

knc_scores = cross_val_score(knc, X, Y, cv = 10)
print("\n")
print(knc_scores)
print(f"\nMean of accuracy scores is for KNN Classifier is {knc_scores.mean()*100}\n")

x = metrics.accuracy_score(y_test, pred_knc)
acc.append(x)
model.append('knc')


# In[64]:


print(classification_report(y_test, pred_knc))


# In[65]:


confusion_matrix(y_test, pred_knc)


# ### 4.SVC

# In[66]:


svc = SVC(kernel = 'rbf')
svc.fit(x_train, y_train)
pred_svc = svc.predict(x_test)
print("Accuracy Score of Support Vector Classifier model is", accuracy_score(y_test, pred_svc)*100)

svc_scores = cross_val_score(svc, X, Y, cv = 10)
print("\n")
print(svc_scores)
print(f"\nMean of accuracy scores for SVC Classifier is {svc_scores.mean()*100}\n")

x = metrics.accuracy_score(y_test, pred_svc)
acc.append(x)
model.append('svc')


# In[67]:


print(classification_report(y_test, pred_svc))


# In[68]:


confusion_matrix(y_test, pred_svc)


# ### 5.Random Forest

# In[69]:


rfc = RandomForestClassifier()

rfc.fit(x_train, y_train)

pred_rfc = rfc.predict(x_test)

print("Accuracy Score of Random Forest model is", accuracy_score(y_test, pred_rfc)*100)

rfc_scores = cross_val_score(rfc, X, Y, cv = 10)
print("\n")
print(rfc_scores)
print(f"\nMean of accuracy scores for Random Forest Classifier is {rfc_scores.mean()*100}\n")

x = metrics.accuracy_score(y_test, pred_rfc)
acc.append(x)
model.append('rfc')


# In[70]:


print(classification_report(y_test, pred_rfc))


# In[71]:


confusion_matrix(y_test, pred_rfc)


# ### 6.MultinominalNB

# In[72]:


nb = MultinomialNB()

nb.fit(x_train, y_train)

pred_nb = nb.predict(x_test)

print("Accuracy Score of MultinomialNB model is", accuracy_score(y_test, pred_nb)*100)

nb_scores = cross_val_score(nb, X, Y, cv = 10)
print("\n")
print(nb_scores)
print(f"\nMean of accuracy scores for MultinomialNB is {nb_scores.mean()*100}\n")

x = metrics.accuracy_score(y_test, pred_nb)
acc.append(x)
model.append('nb')


# In[73]:


print(classification_report(y_test, pred_nb))


# In[74]:


confusion_matrix(y_test, pred_nb)


# ### 7.GaussianNB

# In[75]:


gb = GaussianNB()

gb.fit(x_train,y_train)

pred_gb = gb.predict(x_test)

print("Accuracy of GaussianNB model is",accuracy_score(y_test,pred_gb)*100)

gb_scores = cross_val_score(gb,X,Y,cv = 10)
print("\n")
print(nb_scores)
print(f"\nMean of accureacy score for GaussianNB is {gb_scores.mean()*100}\n")

x = metrics.accuracy_score(y_test, pred_gb)
acc.append(x)
model.append('gb')


# In[76]:


print(classification_report(y_test, pred_gb))


# In[77]:


confusion_matrix(y_test, pred_gb)


# ### 8.Adaboost Classifire

# In[78]:


ada= AdaBoostClassifier()

ada.fit(x_train, y_train)

pred_ada = ada.predict(x_test)

print("Accuracy Score of ADA Boost model is", accuracy_score(y_test, pred_ada)*100)

ada_scores = cross_val_score(ada,X,Y,cv=10)
print("\n")
print(ada_scores)
print(f"\nMean of accureacy score for GaussianNB is {ada_scores.mean()*100}\n")

x = metrics.accuracy_score(y_test, pred_ada)
acc.append(x)
model.append('ada')


# In[79]:


print(classification_report(y_test, pred_ada))


# In[80]:


confusion_matrix(y_test, pred_ada)


# ### 9.Xgboost

# In[81]:


xb = XGBClassifier()

xb.fit(x_train,y_train)

pred_xb = xb.predict(x_test)

print("Accuracy Score of Xgboost model is", accuracy_score(y_test, pred_xb)*100)

xb_score = cross_val_score(xb,X,Y,cv=10)
print("\n")
print(xb_score)
print(f"\nMean of accuracy score for Xgboost is {xb_score.mean()*100}\n")

x = metrics.accuracy_score(y_test, pred_xb)
acc.append(x)
model.append('xb')


# In[82]:


print(classification_report(y_test, pred_xb))


# In[83]:


confusion_matrix(y_test, pred_xb)


# ### 10.ANN
# 

# In[94]:


from tensorflow import keras
from tensorflow.keras import layers
ANN= keras.Sequential()
ANN.add(tf.keras.layers.Dense(20, input_dim=6,  activation='relu')) #1st layer or input layer
ANN.add(tf.keras.layers.Dense(10,  activation='relu')) #2nd layer
ANN.add(tf.keras.layers.Dense(10,  activation='relu')) #3nd layer
ANN.add(tf.keras.layers.Dense(1, activation='sigmoid')) #4rd layer or output layer


# In[95]:


# Compile model
ANN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[96]:


ANN.summary()


# In[97]:


# Fit the model
history = ANN.fit(X, Y, validation_split=0.30, epochs= 250, batch_size=6)


# In[98]:


scores = ANN.evaluate(X, Y)
print( scores[1]*100)


# In[99]:


# Visualize training history

# list all data in history
history.history.keys()


# In[100]:


# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[101]:


x=scores[1]


# In[102]:


acc.append(x)
model.append('ANN')


# ### Accuracy Comparison

# In[103]:


plt.figure(figsize=(25,6))
plt.title("Accuracy Comparision")
plt.xlabel("Accuracy")
plt.ylabel("Algorithm")
sns.barplot(x= acc,y=model)


# In[104]:


accuracy_models = dict(zip(model, acc))
for k, v in accuracy_models.items():
    print (k, '=', v)


# __Choosing ANN and SVC as final models based on their high accuracy.__

# __Hyper Parameter Tuning of SVC__

# In[130]:


from sklearn.model_selection import GridSearchCV

clf = SVC()
param_grid = [{'kernel':['rbf','poly'],'gamma':[50,5,10,0.5],'C':[12,10,0.1,0.001]}]
gsv = GridSearchCV(clf,param_grid,cv=50)
gsv.fit(x_train,y_train)


# In[131]:


gsv.best_estimator_


# In[136]:



svc = SVC(kernel="rbf",C=12,gamma=0.5)
svc.fit(x_train, y_train)
pred_svc = svc.predict(x_test)
print("Accuracy Score of Support Vector Classifier model is", accuracy_score(y_test, pred_svc)*100)


# In[137]:


print(classification_report(y_test, pred_svc))


# In[139]:


confusion_matrix(y_test, pred_svc)


# In[169]:


import matplotlib.pyplot as plt

X=data[:,:2]
y=data[:,6]

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

model = SVC(kernel="rbf",C=12,gamma=0.5)
clf = model.fit(X, y)

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC ')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()


# __Tuning of Hyperparameters for ANN__

# In[173]:


from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')


# In[177]:


# Standardization

X=data[:,0:6]
Y=data[:,6]
a = StandardScaler()
a.fit(X)
X_standardized = a.transform(X)


# In[178]:


# Importing the necessary packages
from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
#from keras.optimizers import Adam
from keras.optimizers import adam_v2
from tensorflow.keras.optimizers import Adam


# In[179]:


# create model
def create_model():
    M1 = Sequential(name='Hypterparameter-Tuning-Dummy')
    M1.add(Dense(12, input_dim=6, kernel_initializer='uniform', activation='relu'))
    M1.add(Dense(8,kernel_initializer='uniform', activation='relu'))
    M1.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    
    adam=Adam(learning_rate=0.01)
    M1.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return M1


# In[180]:


# Create the model
M2 = KerasClassifier(build_fn = create_model,verbose = 0)
# Define the grid search parameters
batch_size = [4,5,6]
epochs = [50,100,150,250]
# Make a dictionary of the grid search parameters
param_grid = dict(batch_size = batch_size,epochs = epochs)
# Build and fit the GridSearchCV
grid = GridSearchCV(estimator = M2,param_grid = param_grid,cv = KFold(),verbose = 10)
grid_result = grid.fit(X_standardized,Y)


# In[181]:


# Summarize the results
print('Best : {}, using {}'.format(grid_result.best_score_,grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
  print('{},{} with: {}'.format(mean, stdev, param))


# In[ ]:




