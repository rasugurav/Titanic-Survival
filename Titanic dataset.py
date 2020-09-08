#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")


# In[3]:


train_data.head()


# In[4]:


test_data.head()


# In[5]:


train_data.info()


# In[6]:


test_data.info()


# In[7]:


median_fare=test_data['Fare'].median()
test_data['Fare'].replace(np.nan,median_fare,inplace=True)


# In[8]:


train_data['Embarked'].unique()


# In[9]:


train_data['Embarked'].mode()


# In[10]:


train_data['Embarked'].replace(np.nan,'S',inplace=True)


# In[11]:


train_data['Embarked'].unique()


# In[12]:


sns.heatmap(test_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[13]:


sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[14]:


train_data.drop('Cabin',axis=1,inplace=True)


# In[15]:


test_data.drop('Cabin',axis=1,inplace=True)


# In[16]:


sns.boxplot(x='Embarked',y='Age',data=train_data)


# In[17]:


def replace_age_train(cols):
    fill_age=train_data[['Age','Embarked']].groupby('Embarked').mean()
    Age=cols[0]
    Embarked=cols[1]
    if pd.isnull(Age):
        return int(fill_age['Age'][Embarked])
    else:
        return Age
def replace_age_test(cols):
    fill_age=test_data[["Age","Embarked"]].groupby("Embarked").mean()
    Age=cols[0]
    Embarked=cols[1]
    if pd.isnull(Age):
        return int(fill_age['Age'][Embarked])
    else:
        return Age


# In[18]:


train_data['Age']=train_data[['Age','Embarked']].apply(replace_age_train,axis=1)


# In[19]:


test_data['Age']=test_data[['Age','Embarked']].apply(replace_age_test,axis=1)


# In[20]:


sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[21]:


sns.heatmap(test_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[22]:


test_data['Age_distribution']=pd.cut(test_data['Age'],bins=[-1,18,25,35,60,100]
                                      ,labels=['minor','young','adult','middleage','senior_citizen'])


# In[23]:


train_data['Age_distribution']=pd.cut(train_data['Age'],bins=[-1,18,25,35,60,100]
                                      ,labels=['minor','young','adult','middleage','senior_citizen'])


# In[24]:


train_data.head()


# In[25]:


test_data.head()


# In[26]:



plt.hist(test_data['Age_distribution'])
plt.xlabel('Age Distribution')
plt.ylabel('Count')
plt.title('Age Distribution Bins')
plt.show()


# In[27]:




plt.hist(train_data['Age_distribution'])
plt.xlabel('Age Distribution')
plt.ylabel('Count')
plt.title('Age Distribution Bins')


# In[28]:



plt.hist(train_data['Embarked'])
plt.xlabel('Embarked')
plt.ylabel('Count')
plt.title('Embarked Bins')


# In[29]:



plt.hist(test_data['Embarked'])
plt.xlabel('Embarked')
plt.ylabel('Count')
plt.title('Embarked Bins')


# In[30]:



colors=['#8ACC17','#C8EC3E', '#FFF412', '#CA0772', '#800561']
labels=['minor','young','adult','middleaged','senior_citizen']
figsize=(10,7)
plt.pie(train_data['Age_distribution'].value_counts(),colors=colors,startangle=90,autopct='%.1f%%',shadow=True,labels=labels)
plt.title('Age distribution')
plt.show()


# In[31]:


colors=['#8ACC17','#C8EC3E', '#FFF412', '#CA0772', '#800561']
labels=['minor','young','adult','middleaged','senior_citizen']
figsize=(10,7)
plt.pie(test_data['Age_distribution'].value_counts(),colors=colors,startangle=90,autopct='%.1f%%',shadow=True,labels=labels)
plt.title('Age Distribution')
plt.show()


# In[32]:


sns.heatmap(train_data.corr(),annot=True,fmt='.2f',cmap='coolwarm')


# In[33]:


figsize=(10,7)
sns.barplot(x='Age_distribution',y='Survived',hue='Sex',data=train_data)


# In[34]:


train_data['Family']=train_data['SibSp']+train_data['Parch']


# In[35]:


train_data.head()


# In[36]:


test_data['Family']=test_data['SibSp']+test_data['Parch']


# In[37]:


test_data.head()


# In[38]:


sns.barplot(x='Family',y='Survived',data=train_data)


# In[39]:


sns.barplot(x='Sex',y='Survived',data=train_data)


# In[ ]:





# In[40]:


train_data.drop(['Ticket','Name'],axis=1,inplace=True)


# In[41]:


train_data.head()


# In[42]:


train_data.drop(['PassengerId'],axis=1,inplace=True)


# In[43]:


train_data.drop(['Fare'],axis=1,inplace=True)


# In[44]:


test_data.head()


# In[45]:



test_data.drop(['Name','Ticket','Fare'],axis=1,inplace=True)


# In[46]:


test_data.head()


# In[47]:


train_data['Sex']=pd.get_dummies(train_data['Sex'])
test_data['Sex']=pd.get_dummies(test_data['Sex'])
train_data['Embarked']=pd.factorize(train_data['Embarked'])[0]
test_data['Embarked']=pd.factorize(test_data['Embarked'])[0]
train_data['Age']=train_data['Age'].astype('int32')
test_data['Age']=test_data['Age'].astype('int32')

def Age_distribution(age):
    if(age <= 16):
        return 0 
    elif age > 16 and age <= 32:
        return 1
    elif age>32 and age <=48:
        return 2 
    elif age>48 and age <= 64:
        return 3
    else:
        return 4
train_data['Age_distribution'] = train_data['Age'].apply(Age_distribution)
def Age_distribution(age):
    if(age <= 16):
        return 0 
    elif age > 16 and age <= 32:
        return 1
    elif age>32 and age <=48:
        return 2 
    elif age>48 and age <= 64:
        return 3
    else:
        return 4
test_data['Age_distribution'] = test_data['Age'].apply(Age_distribution)


# In[48]:


train_data.head()


# In[49]:


test_data.head()


# Multiple Regression Model
# 

# In[50]:


from sklearn import linear_model
regr=linear_model.LinearRegression()


# In[51]:


x=np.asanyarray(train_data[['Age','Family','Embarked']])
y=np.asanyarray(train_data['Survived'])
regr.fit(x,y)
print('coefficients:',regr.coef_)


# Logistic Regression

# In[89]:


from sklearn.model_selection import train_test_split
#all_features = train_data.drop("Survived",axis=1)
#Targeted_feature = train_data["Survived"]
#X_train,X_test,y_train,y_test = train_test_split(all_features,Targeted_feature,test_size=0.3,random_state=42)
#X_train.shape,X_test.shape,y_train.shape,y_test.shape
X = np.array(train_data.drop(['Survived'], 1))
y = np.array(train_data['Survived'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[90]:


from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)
acc_LR = round(logreg.score(X_train, y_train) * 100, 2)
print(acc_LR)
print(confusion_matrix(y_test,Y_pred))
print(classification_report(y_test,Y_pred))


# ## K Nearest Neighbor  

# In[91]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 9) 
knn.fit(X_train, y_train)  
Y_pred = knn.predict(X_test)  
acc_KNN = round(knn.score(X_train, y_train) * 100, 2)
print(acc_KNN)

print(confusion_matrix(y_test,Y_pred))
print(classification_report(y_test,Y_pred))


# # Decision Tree

# In[92]:


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
Y_pred = dtree.predict(X_test)
acc_DT = round(dtree.score(X_train, y_train) * 100, 2)
print(acc_DT)

print(confusion_matrix(y_test,Y_pred))
print(classification_report(y_test,Y_pred))


# # Random Forest

# In[93]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

Y_prediction = random_forest.predict(X_test)
acc_RF = round(random_forest.score(X_train, y_train) * 100, 2)
print(acc_RF)

print(confusion_matrix(y_test,Y_pred))
print(classification_report(y_test,Y_pred))


# # Gaussian Naive Bayes

# In[94]:


from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB() 
gaussian.fit(X_train, y_train)  
Y_pred = gaussian.predict(X_test)
acc_GN = round(gaussian.score(X_train, y_train) * 100, 2)
print(acc_GN)
print(confusion_matrix(y_test,Y_pred))
print(classification_report(y_test,Y_pred))


# In[97]:


models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 
              'Random Forest', 'Decision Tree','Gaussian Naive Bayes'],
    'Score': [acc_RF, acc_KNN, acc_GN, 
              acc_DT, acc_LR]})
models.sort_values(by='Score', ascending=False)


# In[ ]:




