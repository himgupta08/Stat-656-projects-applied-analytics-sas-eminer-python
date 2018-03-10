
# coding: utf-8

# In[1]:


import sys
sys.path.append("D:\\Himanshu\\Acads\\02. Spring semester\\02. Stat 656\\Prof. Jones Modules")

# classes for logistic regression
from Class_regression import logreg
from sklearn.linear_model import LogisticRegression

from sklearn.tree import export_graphviz
from pydotplus.graphviz import graph_from_dot_data
import graphviz
from Class_replace_impute_encode import ReplaceImputeEncode

# classes for neural network
from Class_FNN import NeuralNetwork
from sklearn.neural_network import MLPClassifier

#other needed classes
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy  as np


# In[2]:


file_path = 'D:\\Himanshu\\Acads\\02. Spring semester\\02. Stat 656\\Lectures\\Week-6 Neural Network\Homework\\'
df = pd.read_excel(file_path+"CreditHistory_Clean.xlsx")


# In[3]:


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
#     'purpose':[1,('0','1','2','3','4','5','6','7','8','9','X'),[0,0]],
    'property':[2,(1,2,3,4),[0,0]],
    'checking':[2,(1,2,3,4),[0,0]],
    'telephon':[1,(1,2),[0,0]]}

rie = ReplaceImputeEncode(data_map=attribute_map, drop=False, nominal_encoding='one-hot', display=True, interval_scale='std')
encoded_df = rie.fit_transform(df)


# In[4]:


X=encoded_df.drop('good_bad', axis=1)
Y=encoded_df['good_bad']
np_y = np.ravel(Y)

features = X.columns
classes = ['Good', 'bad']


# In[9]:


print("\n ***** Run Neural network ***** ")

neurons = [(5, 4), (6, 5), (7,6), 3, 11]

for n in neurons:
    fnn = MLPClassifier(hidden_layer_sizes=n, activation='logistic', solver='lbfgs', max_iter=1000, random_state=12345)

    fnn = fnn.fit(X, np_y)
    NeuralNetwork.display_binary_metrics(fnn, X, np_y)

    # Neural Network Cross-Validation
    score_list = ['accuracy', 'recall', 'precision', 'f1']
    mean_score = []
    std_score  = []
    for s in score_list:
        fnn_10 = cross_val_score(fnn, X, np_y, cv=10, scoring=s)
        mean_score.append(fnn_10.mean())
        std_score.append(fnn_10.std())

    print("{:.<13s}{:>6s}{:>13s}".format("\nMetric", "Mean", "Std. Dev."))
    for i in range(len(score_list)):
        score_name = score_list[i]
        mean       = mean_score[i]
        std        = std_score[i]
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(score_name, mean, std))


# In[7]:


# number of weights for single layer with 3 perceptron
# 3*(58 +1 + 1) + 1  (1 is added with 58 due to binary target and b1=1 because hidden layer has bias weight. output layer would also have bias weight so b2=1)

# For layer with (5, 4) perceptron
# 5*(58+1+1)+1 = 301
# ?? not sure how weights are calculated.


# In[10]:


from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state = 100)

fnn = MLPClassifier(hidden_layer_sizes=(5,4), activation='logistic', solver='lbfgs', max_iter=1000, random_state=12345)

np_train_y = np.ravel(train_y)
np_test_y = np.ravel(test_y)

fnn = fnn.fit(train_x, np_train_y)

NeuralNetwork.display_binary_metrics(fnn, test_x, np_test_y)

