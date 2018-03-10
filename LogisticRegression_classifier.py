
# coding: utf-8

# ### Homework 4 code

# #### Import modules written by Prof. Jones

# In[3]:


import sys
sys.path.append("D:\\Himanshu\\Acads\\02. Spring semester\\02. Stat 656\\Prof. Jones Modules")


# #### Import other libraries and load file

# In[10]:


from Class_replace_impute_encode import ReplaceImputeEncode
from Class_regression import logreg
import pandas as pd
import numpy  as np
from sklearn.linear_model import LogisticRegression
import copy as deepcopy

# file_path = 'D:\\Himanshu\\Acads\\02. Spring semester\\02. Stat 656\\Week-4\\HW4\\'
df = pd.read_excel("D:\\Himanshu\\Acads\\02. Spring semester\\02. Stat 656\\Lectures\\Week-4\\HW4\\credithistory_HW2.xlsx")


# #### Make an attribute map and then preprocees the data in next steps

# In[11]:


attribute_map = {
    'age':[0,(1,120),[0,0]],
    'amount':[0,(0,100000),[0,0]], # changed upper limit here
    'duration':[0,(1, 1000),[0,0]], # changed upper limit here
    'good_bad':[1,('good','bad'), [0,0]],
    'coapp':[2,(1,2,3),[0,0]],
    'depends':[1,(1,2),[0,0]],
    'employed':[2,(1,2,3,4,5),[0,0]],
    'history':[2,(0,1,2,3,4),[0,0]],
    'existcr':[2,(1,2,3,4),[0,0]],
    'installp':[2,(1,2,3,4),[0,0]],
    'job':[2,(1,2,3,4),[0,0]],
    'housing':[2,(1,2,3),[0,0]],
    'foreign':[1,(1,2),[0,0]],
    'marital':[2,(1,2,3,4),[0,0]],
    'resident':[2,(1,2,3,4),[0,0]],
    'savings':[2,(1,2,3,4,5),[0,0]],
    'other':[2,(1,2,3),[0,0]],
    'property':[2,(1,2,3,4),[0,0]],
    'checking':[2,(1,2,3,4),[0,0]],
    'telephon':[2,(1,2),[0,0]]}


# Next, we need to preprocess these data. They do contain a few outliers and missing values.

# In[12]:


rie = ReplaceImputeEncode(data_map=attribute_map, display=True)
encoded_df = rie.fit_transform(df)


# In[9]:


print(encoded_df['good_bad'].value_counts())
# Bad is encoded as 1 and Good is encoded as -1

# -1.0    700
#  1.0    300
# Name: good_bad, dtype: int64


# In[124]:


# print(encoded_df['good_bad'].value_counts())
# Bad is encoded as -1 and Good is encoded as 1

# print(encoded_df['resident2'].value_counts()) #what is the meaning of 0, -1 in this column?
# encoded_df.columns # Purpose has been dropped
# Same can be get by rie.col command

# print(encoded_df['age'].value_counts()) # Imputed and replaced

mean1 = (encoded_df['age'].mean(), encoded_df['amount'].mean(),encoded_df['duration'].mean())
min1 = (encoded_df['age'].min(), encoded_df['amount'].min(),encoded_df['duration'].min())
max1 = (encoded_df['age'].max(), encoded_df['amount'].max(),encoded_df['duration'].max())
print(mean1, min1, max1)


# In[138]:


var_list = ['employed0', 'employed1', 'employed2', 'employed3', 'marital0', 'marital1', 'marital2',  'savings0', 'savings1', 'savings2', 'savings3']

for i in var_list:
    print(encoded_df[i].value_counts())


# In[13]:


# Regression requires numpy arrays containing all numeric values
y = np.asarray(encoded_df['good_bad']) 
# Drop the target, 'object'.  Axis=1 indicates the drop is for a column.
X = np.asarray(encoded_df.drop('good_bad', axis=1)) 


# ### Fit a logistic regression model

# In[14]:


lgr = LogisticRegression(C=5)
lgr.fit(X,y)


# ### Select best features using RFE feature selection

# In[42]:


from sklearn.feature_selection import RFE
selector = RFE(lgr, 20)
selector.fit_transform(X, y)

ranks = selector.ranking_
X_names = encoded_df.columns.drop('good_bad')
# print sorted(map(lambda x: round(x, 4), selector.ranking_), names)


# In[110]:


rfe_features = np.column_stack((X_names, ranks))
rfe_cols = rfe_features[np.where(rfe_features[:,1]<10),:2][0]
rfe_col1 = rfe_cols[:,:1]
print(rfe_col1)


# ## Run logistic on whole dataset

# In[86]:


logreg.display_coef(lgr, X, y, rie.col)


# In[22]:


logreg.display_binary_metrics(lgr, X, y)


# ### Run Logistic Regression on train and test
# 

# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate =             train_test_split(X,y,test_size = 0.3, random_state=7)
lgr_train = LogisticRegression()
lgr_train.fit(X_train, y_train)
print("\nTraining Data\nRandom Selection of 70% of Original Data")
logreg.display_binary_split_metrics(lgr_train, X_train, y_train,                                     X_validate, y_validate)


# ## K fold crossvalidation

# In[24]:


from sklearn.model_selection import cross_val_score
lgr_4_scores = cross_val_score(lgr, X_train, y_train, cv=4)
print("\nAccuracy Scores by Fold: ", lgr_4_scores)
print("Accuracy Mean:      %.4f" %lgr_4_scores.mean())
print("Accuracy Std. Dev.: %.4f" %lgr_4_scores.std())

max(lgr_4_scores)

