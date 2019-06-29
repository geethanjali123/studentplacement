
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
#reading student data into a dataframe from csv file
df=pd.read_csv('updatedstudentscrtdata.csv')
#removing unnecessary columns
print(df.columns)
df=df.drop(df.columns[17:72],axis=1)
#label encoder is to convert categorical values to numeric values
n=LabelEncoder()
df['Branch']=n.fit_transform(df['Branch'].astype('str'))
df['Gender']=n.fit_transform(df['Gender'].astype('str'))


# In[3]:


df=df.drop(['RollNumber','Name','Category','Scholarship'],axis=1)
#Filling NAN values with 0
for x in df.columns:
    df[x]=df[x].fillna(0)
sns.pairplot(df)


# In[4]:


#removing columns which has correlation based on pairplot
df=df.drop(['X1st.year','X2nd.year','X3rd.year','X4th.four'],axis=1)


# In[5]:


def comgrade(df):
    df['D1']=0
    for i,r in df.iterrows():
        if r['A']!=0:
             df.loc[i,'D1']=1
        elif r['B']!=0: 
             df.loc[i,'D1']=2
        elif r['C']!=0:
             df.loc[i,'D1']=3  


# In[6]:


#function to normalize
def nor(f):
    f.loc[:,'TenthMarks']/=600
    f.loc[:,'InterMarks']/=1000
    f.loc[:,'Aggregate']/=100
    f.loc[:,'EamcetRank']/=10000
    return f
df=nor(df)    


# In[7]:


#features and target
def featar(df):
    df['D']=0
    for i,r in df.iterrows():
        if r['A']==0 and r['B']==0 and r['C']==0:
             df.loc[i,'D']=0
        else:
             df.loc[i,'D']=1  
    comgrade(df)
    return df['D1'],df.drop(['A','B','C','D1'],axis=1)
y,X=featar(df)
X


# In[128]:


#logistic classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.6,test_size=0.4)
logreg=LogisticRegression(solver='lbfgs',multi_class='multinomial')
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred)*100)


# In[28]:


#naive bayes algorithm
from sklearn.naive_bayes import GaussianNB
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.6,test_size=0.4)
gnd=GaussianNB()
gnd.fit(X_train,y_train)
y_pred=gnd.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred)*100)


# In[38]:


from sklearn import ensemble
from sklearn.metrics import r2_score
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,train_size=0.6)
params = {'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 2,
          'learning_rate': 0.01}
clf = ensemble.GradientBoostingClassifier()
clf.fit(X_train, y_train)
print(metrics.accuracy_score(y_test,clf.predict(X_test)))


# In[49]:


from sklearn import tree
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,train_size=0.6)
clf=tree.DecisionTreeClassifier()
clf=clf.fit(X_train,y_train)
print(metrics.accuracy_score(y_test,clf.predict(X_test))*100)


# In[174]:


import matplotlib.pyplot as plt
from sklearn import svm
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,train_size=0.6)

clf = svm.SVC(kernel='linear', C = 1.0)

clf.fit(X_train,y_train)
l=clf.predict(X_test)
print(metrics.accuracy_score(y_test,l)*100)
plt.hist(l,bins=[1,2,3])

