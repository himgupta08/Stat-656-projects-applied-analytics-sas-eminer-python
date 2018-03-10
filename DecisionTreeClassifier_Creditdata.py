
# coding: utf-8

# In[2]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score
import graphviz as show_tree
from pydotplus import graph_from_dot_data
import pandas as pd

import sys
sys.path.append("D:\\Himanshu\\Acads\\02. Spring semester\\02. Stat 656\\Prof. Jones Modules")
from Class_tree import DecisionTree
from Class_replace_impute_encode import ReplaceImputeEncode


# In[3]:


file_path = 'D:\\Himanshu\\Acads\\02. Spring semester\\02. Stat 656\\Lectures\\Week-5 Decision tree\\Homework 5\\'
df = pd.read_excel(file_path+"CreditHistory_Clean.xlsx")


# In[4]:


# cat_cols = ['checking','history', 'purpose','savings', 'employed', 'installp', 'marital', 'coapp', 'resident',
#        'property', 'other', 'housing', 'existcr', 'job', 'depends', 'telephon','foreign', 'good_bad']
# for col in cat_cols:
#     df[col] = df[col].astype('str')

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

rie = ReplaceImputeEncode(data_map=attribute_map, display=True)
encoded_df = rie.fit_transform(df)


# In[5]:


from collections import Counter
Counter(encoded_df['employed0'])
len(encoded_df.columns) # 46 columns

Counter(encoded_df['good_bad'])


# In[6]:


X = encoded_df.drop('good_bad', axis=1)
y = encoded_df['good_bad']


# In[7]:


features = X.columns
classes = ['Good', 'bad']


# ### For decision tree of depth 5
# 

# In[29]:


dtc_depth5 = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=5, min_samples_leaf=5)
dtc_depth5 = dtc_depth5.fit(X,y)

DecisionTree.display_importance(dtc_depth5, features)
DecisionTree.display_binary_metrics(dtc_depth5, X, y)
dot_data_depth5 = export_graphviz(dtc_depth5, filled=True, rounded=True, class_names=classes, feature_names = features, out_file=None)

# Accuracy =  0.7780


# In[31]:


#write tree to png file 
graph_png = graph_from_dot_data(dot_data_depth5)

graph_path = 'D:\\Himanshu\\Acads\\02. Spring semester\\02. Stat 656\\Lectures\\Week-5 Decision tree\\'
graph_png.write_png(graph_path+'credithistory_depth51.png')


# In[27]:



score_list = ['accuracy', 'recall', 'precision', 'f1']
mean_score = []
std_score = []
print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))

for s in score_list:
    dtc_10 = cross_val_score(dtc_depth5, X, y, scoring=s, cv=10)
    mean = dtc_10.mean()
    std = dtc_10.std()
    mean_score.append(mean)
    std_score.append(std)
    print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
    
# Accuracy - 0.7050, F1 - 0.8038 


# ### Running Decision tree for Depth 6

# In[32]:


dtc_depth6 = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_split=5, min_samples_leaf=5)
dtc_depth6 = dtc_depth6.fit(X,y)

# DecisionTree.display_importance(dtc_depth6, features)
DecisionTree.display_binary_metrics(dtc_depth6, X, y)
dot_data_depth6 = export_graphviz(dtc_depth6, filled=True, rounded=True, class_names=classes, feature_names = features, out_file=None)

# Accuracy = 0.8060


# In[11]:


#write tree to png file 
graph_png = graph_from_dot_data(dot_data_depth6)

graph_path = 'D:\\Himanshu\\Acads\\02. Spring semester\\02. Stat 656\\Lectures\\Week-5 Decision tree\\'
graph_png.write_png(graph_path+'credithistory_depth6.png')

score_list = ['accuracy', 'recall', 'precision', 'f1']
mean_score = []
std_score = []
print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))

for s in score_list:
    dtc_10 = cross_val_score(dtc_depth6, X, y, scoring=s, cv=10)
    mean = dtc_10.mean()
    std = dtc_10.std()
    mean_score.append(mean)
    std_score.append(std)
    print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))

# Accuracy - 0.7060, F1 = 0.7986


# ### Decision tree for depth 12

# In[33]:


dtc_depth12 = DecisionTreeClassifier(criterion='gini', max_depth=12, min_samples_split=5, min_samples_leaf=5)
dtc_depth12 = dtc_depth12.fit(X,y)

# DecisionTree.display_importance(dtc_depth12, features)
DecisionTree.display_binary_metrics(dtc_depth12, X, y)
dot_data_depth12 = export_graphviz(dtc_depth12, filled=True, rounded=True, class_names=classes, feature_names = features, out_file=None)

# Accuracy = 0.8710


# In[13]:


#write tree to png file 
graph_png = graph_from_dot_data(dot_data_depth12)

graph_path = 'D:\\Himanshu\\Acads\\02. Spring semester\\02. Stat 656\\Lectures\\Week-5 Decision tree\\'
graph_png.write_png(graph_path+'credithistory_depth12.png')

score_list = ['accuracy', 'recall', 'precision', 'f1']
mean_score = []
std_score = []
print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))

for s in score_list:
    dtc_10 = cross_val_score(dtc_depth12, X, y, scoring=s, cv=10)
    mean = dtc_10.mean()
    std = dtc_10.std()
    mean_score.append(mean)
    std_score.append(std)
    print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))

# accuracy - 0.6750 , F1 score = 0.7629 


# ### Decision tree for depth 10

# In[34]:


dtc_depth10 = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=5, min_samples_leaf=5)
dtc_depth10 = dtc_depth10.fit(X,y)

# DecisionTree.display_importance(dtc_depth10, features)
DecisionTree.display_binary_metrics(dtc_depth10, X, y)
dot_data_depth10 = export_graphviz(dtc_depth10, filled=True, rounded=True, class_names=classes, feature_names = features, out_file=None)

# Accuracy = 0.8590

score_list = ['accuracy', 'recall', 'precision', 'f1']
mean_score = []
std_score = []
print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))

for s in score_list:
    dtc_10 = cross_val_score(dtc_depth10, X, y, scoring=s, cv=10)
    mean = dtc_10.mean()
    std = dtc_10.std()
    mean_score.append(mean)
    std_score.append(std)
    print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
    
# Accuracy - 0.6900, F1 score = 0.7719


# ### Decision tree for depth 8

# In[35]:


dtc_depth8 = DecisionTreeClassifier(criterion='gini', max_depth=8, min_samples_split=5, min_samples_leaf=5)
dtc_depth8 = dtc_depth8.fit(X,y)

# DecisionTree.display_importance(dtc_depth8, features)
DecisionTree.display_binary_metrics(dtc_depth8, X, y)
dot_data_depth8 = export_graphviz(dtc_depth8, filled=True, rounded=True, class_names=classes, feature_names = features, out_file=None)

# Accuracy = 0.8320

score_list = ['accuracy', 'recall', 'precision', 'f1']
mean_score = []
std_score = []
print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))

for s in score_list:
    dtc_10 = cross_val_score(dtc_depth8, X, y, scoring=s, cv=10)
    mean = dtc_10.mean()
    std = dtc_10.std()
    mean_score.append(mean)
    std_score.append(std)
    print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
    
# Accuracy 0.6840, F1 =  0.7692 


# ### Clearly Depth 5 has best accuracy and F1 score in trees above
# ### Running Decision tree of depth 5 with Entropy to see any change

# In[37]:


# Depth 5 has the highest accuracy - try entropy criteria

dtc_depth5_1 = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=5, min_samples_leaf=5, max_features=None)
dtc_depth5_1 = dtc_depth12.fit(X,y)

# DecisionTree.display_importance(dtc_depth5_1, features)
DecisionTree.display_binary_metrics(dtc_depth5_1, X, y)
dot_data_depth5_1 = export_graphviz(dtc_depth5_1, filled=True, rounded=True, class_names=classes, feature_names = features, out_file=None)

# Accuracy = 0.8800

score_list = ['accuracy', 'recall', 'precision', 'f1']
mean_score = []
std_score = []
print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))

for s in score_list:
    dtc_10 = cross_val_score(dtc_depth5_1, X, y, scoring=s, cv=10)
    mean = dtc_10.mean()
    std = dtc_10.std()
    mean_score.append(mean)
    std_score.append(std)
    print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))

# Accuracy is quite low in this case. so gini is better


# 
# ## Creating train test split and using decision tree with depth 5 (gini criteria)
# 

# In[23]:


from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state = 100)


# In[24]:


dtc = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf=5, min_samples_split=5, max_features=None)
dtc = dtc.fit(train_x, train_y)

DecisionTree.display_importance(dtc, features)
DecisionTree.display_binary_metrics(dtc, train_x, train_y)

# Accuracy = 0.7943


# In[25]:


predict_test_y = dtc.predict(test_x)


# In[26]:


DecisionTree.display_binary_metrics(dtc, test_x, test_y)
# accuracy - 0.7033, F1 score=0.8000

