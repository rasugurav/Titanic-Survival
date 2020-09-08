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


# In[40]:


test_data.info()


# In[41]:


train_data.drop(['Ticket','Name'],axis=1,inplace=True)


# In[42]:


train_data.head()


# In[43]:


#train_data.drop(['PassengerId'],axis=1,inplace=True)


# In[44]:


train_data.drop(['Fare'],axis=1,inplace=True)


# In[45]:


test_data.head()


# In[46]:



test_data.drop(['Name','Ticket','Fare'],axis=1,inplace=True)


# In[47]:


test_data.head()


# In[48]:


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


# In[49]:


train_data.head()


# In[50]:


test_data['PassengerId']


# Logistic Regression

# In[51]:


X =train_data.drop(['Survived'],axis=1)
y =train_data['Survived']


# In[52]:




train_data.set_index('PassengerId',inplace=True)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
leb=LabelEncoder()
test_data.set_index('PassengerId',inplace=True)
from sklearn.preprocessing import LabelEncoder
let=LabelEncoder()
lebt=LabelEncoder()


# In[53]:


test_data.head()


# In[54]:



le.fit(train_data['Sex'])
leb.fit(train_data["Embarked"].astype(str))
#let.fit(test_data['Sex'])
#lebt.fit(test_data["Embarked"].astype(str))


# In[55]:




sex=le.transform(train_data['Sex'])
emb=leb.transform(train_data['Embarked'].astype(str))
sext=le.transform(test_data['Sex'])
embt=leb.transform(test_data['Embarked'].astype(str))


# In[56]:




train_data['Sex']=sex
train_data['Embarked']=emb
test_data['Sex']=sext
test_data['Embarked']=embt


# In[57]:


X =train_data.drop(['Survived'],axis=1)
y =train_data['Survived']


# In[58]:


test_data.info()


# In[59]:


from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

logreg = LogisticRegression()
logreg.fit(X,y)

acc_LR = round(logreg.score(X, y) * 100, 2)
print(acc_LR)


# In[60]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 9) 
knn.fit(X, y)  

acc_KNN = round(knn.score(X, y) * 100, 2)
print(acc_KNN)


# In[61]:


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X,y)
acc_DT = round(dtree.score(X, y) * 100, 2)
print(acc_DT)


# In[62]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X, y)

acc_RF = round(random_forest.score(X, y) * 100, 2)
print(acc_RF)


# In[63]:


from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB() 
gaussian.fit(X, y)  

acc_GN = round(gaussian.score(X, y) * 100, 2)
print(acc_GN)


# In[64]:


models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 
              'Random Forest', 'Decision Tree','Gaussian Naive Bayes'],
    'Score': [acc_RF, acc_KNN, acc_GN, 
              acc_DT, acc_LR]})
models.sort_values(by='Score', ascending=False)


# In[65]:



predict=knn.predict(test_data)
predict


# In[67]:



data={'PassengerId':test_data.index,'Survived':predict}    

df=pd.DataFrame(data)


# In[68]:


df.to_csv('prediction.csv',index=False)


# In[70]:


df1=pd.read_csv('prediction.csv')
df1


# In[ ]:





# In[ ]:




