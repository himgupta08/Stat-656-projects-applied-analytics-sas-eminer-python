#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 23:40:14 2018

@author: deepthisen
"""

#STAT 656 MIDTERM COMBINED CODE - SUBMITTED BY GROUP 12
#Please use the Class versions (Class_tree, Class_FNN etc.)
#provided along with the submission
print("***********************************************************")
print("DECISION TREE MODEL \n")
print("***********************************************************")
#Import all required classes and libraries
from Class_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
from pydotplus import graph_from_dot_data
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from Class_replace_impute_encode import ReplaceImputeEncode
import graphviz
import pandas as pd
import numpy as np


file_path = '/Users/deepthisen/Desktop/Courses/stat656/MidTerm Exam/'
df = pd.read_excel(file_path+"CreditCard_Defaults.xlsx")
df=df.drop(['Customer'],axis=1)
print("Authentication Data with %i observations & %i attributes.\n" %df.shape, df[0:2])

attribute_map = {
    'Gender':[1,(1,2),[0,0]], 
    'Education':[2,(0,1,2,3,4,5,6),[0,0]],
    'Marital_Status' :[2,(0,1,2,3),[0,0]],         
    'card_class':[2,(1,2,3),[0,0]],       
    'Age'  :[0,(20,80),[0,0]], 
    'Credit_Limit':[0,(100,80000),[0,0]], 
    'Jun_Status':[0,(-2,8),[0,0]],
    'May_Status':[0,(-2,8),[0,0]],
    'Apr_Status':[0,(-2,8),[0,0]], 
    'Mar_Status':[0,(-2,8),[0,0]],
    'Feb_Status':[0,(-2,8),[0,0]],
    'Jan_Status':[0,(-2,8),[0,0]],
    'Jun_Bill':[0,(-12000,32000),[0,0]],
    'May_Bill':[0,(-12000,32000),[0,0]],
    'Apr_Bill':[0,(-12000,32000),[0,0]], 
    'Mar_Bill':[0,(-12000,32000),[0,0]],
    'Feb_Bill':[0,(-12000,32000),[0,0]],
    'Jan_Bill':[0,(-12000,32000),[0,0]],
    'Jun_Payment':[0,(0,60000),[0,0]],
    'May_Payment':[0,(0,60000),[0,0]],
    'Apr_Payment':[0,(0,60000),[0,0]], 
    'Mar_Payment':[0,(0,60000),[0,0]],
    'Feb_Payment':[0,(0,60000),[0,0]],
    'Jan_Payment':[0,(0,60000),[0,0]],
    'Jun_PayPercent':[0,(0,1),[0,0]],
    'May_PayPercent':[0,(0,1),[0,0]],
    'Apr_PayPercent':[0,(0,1),[0,0]], 
    'Mar_PayPercent':[0,(0,1),[0,0]],
    'Feb_PayPercent':[0,(0,1),[0,0]],
    'Jan_PayPercent':[0,(0,1),[0,0]],
    'Default':[3,(0,1),[0,0]]}

encoding = 'one-hot'# Categorical encoding:  Use 'SAS', 'one-hot' or None
scale    = None     # Interval scaling:  Use 'std', 'robust' or None
# Instantiate the class 
rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding=encoding, \
                          interval_scale = scale, drop=False, display=False)
# Now request replace-impute-encode for dataframe
encoded_df = rie.fit_transform(df)
varlist = ['Default']
X = encoded_df
y = df[varlist]
maxdep_list=[4,5,6,7,8,9]
min_samp_leaf_list=[3]
min_samp_split_list=[3]
score_list = ['accuracy', 'recall', 'precision', 'f1']

for maxdep in maxdep_list:
    for min_leaf in min_samp_leaf_list:
        for min_split in min_samp_split_list:
                dtc = DecisionTreeClassifier(criterion='gini', max_depth=maxdep, \
                                 min_samples_split=min_split, min_samples_leaf=min_leaf)
                dtc = dtc.fit(X,y)
                mean_score = []
                std_score = []
                print("Max depth : %s" % maxdep)
                print("Min sample in leaf : %s" % min_leaf)
                print("Min sample for split : %s" % min_split)
                print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
                for s in score_list:
                    dtc_10 = cross_val_score(dtc, X, y, scoring=s, cv=10)
                    mean = dtc_10.mean()
                    std = dtc_10.std()
                    mean_score.append(mean)
                    std_score.append(std)
                    print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
#On inspecting the cross-validation results, the tree with 7 max depth is seen 
#to offer best performance                    
bestdepth=7
bestminleaf=4
bestminsplit=4
print("Best Max Depth Setting: %s" %bestdepth)
print("Best Min Leaf Sample Setting: %s" %bestminleaf)
print("Best Min Split Sample Setting: %s" %bestminsplit)
print("\n")
print("Splitting the dataset and comparing Training and Validation:")
X_train, X_validate, y_train, y_validate = \
            train_test_split(X,y,test_size = 0.3, random_state=7)
            
#Fitting a tree with the best depth using the training data
dtc1 = DecisionTreeClassifier(criterion='entropy', max_depth=bestdepth, \
min_samples_split=bestminsplit, min_samples_leaf=bestminleaf)
dtc1 = dtc1.fit(X_train,y_train)
features = list(encoded_df)
classes = ['No Default', 'Default']
print("*****RESULTS*******")
DecisionTree.display_importance(dtc1, features)
DecisionTree.display_binary_metrics(dtc1, X_validate, y_validate)
print("*****END OF RESULTS*******")

print("*****SENSITIVITY ENHANCEMENT FOR DECISION TREES*******")
#Finding the best probability threshold by looping over training data and maximizing
#the avg of sensitivity and specificity
metriclist=[]
bestmetric=0
bestthreshold=0
predict_prob=dtc1.predict_proba(X_train)
for i in range(0,100):
    threshold=(i/100)
    predict_mine=np.where(predict_prob<threshold,0,1)
    k=confusion_matrix(y_train,predict_mine[:,1])
    true_neg=k[0,0]
    true_pos=k[1,1]
    true_pred=true_pos+true_neg
    fals_neg=k[1,0]
    fals_pos=k[0,1]
    totalpred=true_pred+fals_pos+fals_neg
    accu=true_pred/totalpred
    sens=true_pos/(true_pos+fals_neg)
    spec=true_neg/(true_neg+fals_pos)
    metric=(sens+spec)/2
    if metric>bestmetric:
        bestmetric=metric
        bestthreshold=threshold
    metriclist.append(metric)


#Printing results for training data using best probability threshold
predict_mine=np.where(predict_prob<bestthreshold,0,1)
k=confusion_matrix(y_train,predict_mine[:,1])
true_neg=k[0,0]
true_pos=k[1,1]
true_pred=true_pos+true_neg
fals_neg=k[1,0]
fals_pos=k[0,1]
totalpred=true_pred+fals_pos+fals_neg
accu=true_pred/totalpred
sens=true_pos/(true_pos+fals_neg)
spec=true_neg/(true_neg+fals_pos)
print("Confusion Matrix AFTER SENSITIVITY ENHANCEMENT: Training")
print(k)
print("Training Accuracy:  %s" %accu)
print("Training Sensitivity:  %s" %sens)
print("Training Specificity:  %s" %spec)

##Printing results for validation data using best probability threshold
predict_prob_val=dtc1.predict_proba(X_validate)
predict_mine_val=np.where(predict_prob_val<bestthreshold,0,1)
k=confusion_matrix(y_validate,predict_mine_val[:,1])
true_neg=k[0,0]
true_pos=k[1,1]
true_pred=true_pos+true_neg
fals_neg=k[1,0]
fals_pos=k[0,1]
totalpred=true_pred+fals_pos+fals_neg
accu=true_pred/totalpred
sens=true_pos/(true_pos+fals_neg)
spec=true_neg/(true_neg+fals_pos)
print("Confusion Matrix AFTER SENSITIVITY ENHANCEMENT: Validation")
print(k)
print("Validation Accuracy:  %s" %accu)
print("Validation Sensitivity:  %s" %sens)
print("Validation Specificity:  %s" %spec)

dot_data = export_graphviz(dtc1, filled=True, rounded=True, \
class_names=classes, feature_names = features, out_file=None)
graph_png = graph_from_dot_data(dot_data)
graph_path = '/Users/deepthisen/Desktop/Courses/stat656/MidTerm Exam/graphs/'
graph_png.write_png(graph_path+'bestfitDTC.png')
graph_pdf = graphviz.Source(dot_data)
graph_pdf.view('bestfitDTC') #Displays tree

print("***********************************************************")
print("END OF DECISION TREE MODEL \n")
print("***********************************************************")

print("\n\n")

print("***********************************************************")
print("LOGISTIC REGRESSION MODEL \n")
print("***********************************************************")


import matplotlib.pyplot as pt
from sklearn.model_selection import cross_val_score
from Class_regression import logreg
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
            
from sklearn.model_selection import cross_validate
df = pd.read_excel(file_path+"CreditCard_Defaults.xlsx")
    
df.dtypes

    
attribute_map = {
        'Default':[1,(0,1),[0,0]],
        'Gender':[1,(1, 2),[0,0]],
        'Education':[2,(0,1,2,3,4,5,6),[0,0]], 
        'Marital_Status':[2,(0,1,2,3), [0,0]],
        'card_class':[2,(1,2,3),[0,0]],
        'Age':[0,(20,80),[0,0]],
        'Credit_Limit':[0,(100, 80000),[0,0]],
        'Jun_Status':[0,(-2, 8),[0,0]],
        'May_Status':[0,(-2, 8),[0,0]],
        'Apr_Status':[0,(-2, 8),[0,0]],
        'Mar_Status':[0,(-2, 8),[0,0]],
        'Feb_Status':[0,(-2, 8),[0,0]],
        'Jan_Status':[0,(-2, 8),[0,0]],
        'Jun_Bill':[0,(-12000, 32000),[0,0]],
        'May_Bill':[0,(-12000, 32000),[0,0]],
        'Apr_Bill':[0,(-12000, 32000),[0,0]],
        'Mar_Bill':[0,(-12000, 32000),[0,0]],
        'Feb_Bill':[0,(-12000, 32000),[0,0]],
        'Jan_Bill':[0,(-12000, 32000),[0,0]],
        'Jun_Payment':[0,(0, 60000),[0,0]],
        'May_Payment':[0,(0, 60000),[0,0]],
        'Apr_Payment':[0,(0, 60000),[0,0]],
        'Mar_Payment':[0,(0, 60000),[0,0]],
        'Feb_Payment':[0,(0, 60000),[0,0]],
        'Jan_Payment':[0,(0, 60000),[0,0]],
        'Jun_PayPercent':[0,(0, 1),[0,0]],
        'May_PayPercent':[0,(0, 1),[0,0]],
        'Apr_PayPercent':[0,(0, 1),[0,0]],
        'Mar_PayPercent':[0,(0, 1),[0,0]],
        'Feb_PayPercent':[0,(0, 1),[0,0]],
        'Jan_PayPercent':[0,(0, 1),[0,0]]
    }
    
    
df.drop(['Customer'], axis=1)
np.sum(df['Marital_Status']==0)
df.dtypes
    
    
rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', display=True, drop = True)
encoded_df = rie.fit_transform(df)
    
    
varlist = df['Default']
X = encoded_df.drop('Default', axis=1)
y = encoded_df['Default']

lgr = LogisticRegression()

#Selecting the best attributes using RFE - 25 attributes chosen   
rfe = RFE(lgr,25)
rfe = rfe.fit(X,y)

#print the attributes with ranking. Attributes with ranks 1 are chosen by the process
print(rfe.support_)
print(rfe.ranking_)


#Code to pick the chosen 25 attribute names and assign it to an array    
a = rfe.ranking_
b = X.columns.values 
c = np.c_[a,b] 
c = pd.DataFrame(c)
c.columns = ['c1','c2']
cols = c.loc[c['c1']== 1,'c2']

#Subset of original dataframe - after RFE
X = X[cols]

#Fitting a LgR Classifier with the best C and penalty parameter using the training data

c_space = np.logspace(-2, 8, 15)
param_grid = ['l1', 'l2']
score_list = ['accuracy', 'recall', 'precision', 'f1']
#max_f1 = 0
for e in c_space:
    for f in param_grid:
        print("C: ", e, "Regularization method: ", f)
        lgr_CV = LogisticRegression(C=e,penalty=f,random_state=12345)
        lgr_CV= lgr_CV.fit(X, y)
        scores = cross_validate(lgr_CV, X, y, scoring=score_list,cv=10)
        
        print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
        for s in score_list:
            var = "test_"+s
            mean = scores[var].mean()
            std  = scores[var].std()
            print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
            

#Printing model metrics for test data
print("Splitting the dataset and comparing Training and Validation:")

X_train, X_validate, y_train, y_validate = \
            train_test_split(X,y,test_size = 0.3, random_state=7)

lgr_train = LogisticRegression(C = 3727593.720314938,penalty='l2',random_state=12345)
lgr_train = lgr_train.fit(X_train, y_train)
#logreg.display_binary_split_metrics(lgr_train, X_train, y_train, X_validate, y_validate)

predict_train = lgr_train.predict(X_train)
predict_validate = lgr_train.predict(X_validate)

conf_matt = confusion_matrix(y_true=y_train, y_pred=predict_train)
conf_matv = confusion_matrix(y_true=y_validate, y_pred=predict_validate)
print("\n")
print("{:.<23s}{:>15s}{:>15s}".format('Model Metrics', \
                                      'Training', 'Validation'))
acct = accuracy_score(y_train, predict_train)
accv = accuracy_score(y_validate, predict_validate)

        
print("{:.<23s}{:15.4f}{:15.4f}".format('Accuracy', acct, accv))
        
print("{:.<23s}{:15.4f}{:15.4f}".format('Precision', \
                      precision_score(y_train,predict_train), \
                      precision_score(y_validate,predict_validate)))
print("{:.<23s}{:15.4f}{:15.4f}".format('Recall (Sensitivity)', \
                      recall_score(y_train,predict_train), \
                      recall_score(y_validate,predict_validate)))
print("{:.<23s}{:15.4f}{:15.4f}".format('F1-score', \
                      f1_score(y_train,predict_train), \
                      f1_score(y_validate,predict_validate)))

######################################################################################
print("********ENHANCEMENTS TO BASE MODEL******")

#Finding the best probability threshold by looping over training data and maximizing
#the avg of sensitivity and specificity
metriclist=[]
bestmetric=0
bestthreshold=0
predict_prob=lgr_train.predict_proba(X_train)
for i in range(0,100):
    threshold=(i/100)
    predict_mine=np.where(predict_prob<threshold,0,1)
    k=confusion_matrix(y_train,predict_mine[:,1])
    true_neg=k[0,0]
    true_pos=k[1,1]
    true_pred=true_pos+true_neg
    fals_neg=k[1,0]
    fals_pos=k[0,1]
    totalpred=true_pred+fals_pos+fals_neg
    accu=true_pred/totalpred
    sens=true_pos/(true_pos+fals_neg)
    spec=true_neg/(true_neg+fals_pos)
    metric=(sens+spec)/2
    if metric>bestmetric:
        bestmetric=metric
        bestthreshold=threshold
    metriclist.append(metric)
    
#Printing results for training data using best probability threshold
predict_mine=np.where(predict_prob<bestthreshold,0,1)
k=confusion_matrix(y_train,predict_mine[:,1])
true_neg=k[0,0]
true_pos=k[1,1]
true_pred=true_pos+true_neg
fals_neg=k[1,0]
fals_pos=k[0,1]
totalpred=true_pred+fals_pos+fals_neg
accu=true_pred/totalpred
sens=true_pos/(true_pos+fals_neg)
spec=true_neg/(true_neg+fals_pos)
print("Confusion Matrix AFTER SENSITIVITY ENHANCEMENT : Training")
print(k)
print("Training Accuracy:  %s" %accu)
print("Training Sensitivity:  %s" %sens)
print("Training Specificity:  %s" %spec)

##Printing results for validation data using best probability threshold
predict_prob_val=lgr_train.predict_proba(X_validate)
predict_mine_val=np.where(predict_prob_val<bestthreshold,0,1)
k=confusion_matrix(y_validate,predict_mine_val[:,1])
true_neg=k[0,0]
true_pos=k[1,1]
true_pred=true_pos+true_neg
fals_neg=k[1,0]
fals_pos=k[0,1]
totalpred=true_pred+fals_pos+fals_neg
accu=true_pred/totalpred
sens=true_pos/(true_pos+fals_neg)
spec=true_neg/(true_neg+fals_pos)
print("Confusion Matrix AFTER SENSITIVITY ENHANCEMENT: Validation")
print(k)
print("Validation Accuracy:  %s" %accu)
print("Validation Sensitivity:  %s" %sens)
print("Validation Specificity:  %s" %spec)

print("***********************************************************")
print("END OF LOGISTIC REGRESSION MODEL \n")
print("***********************************************************")


print("\n\n")

print("***********************************************************")
print("NEURAL NETWORK MODEL \n")
print("***********************************************************")

# classes for neural network

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

import pandas as pd
import numpy as np

import sys
sys.path.append("/Users/deepthisen/Desktop/Courses/stat656/MidTerm Exam/")

from Class_replace_impute_encode import ReplaceImputeEncode
from Class_FNN import NeuralNetwork

# Load dataset
file_path = '/Users/deepthisen/Desktop/Courses/stat656/MidTerm Exam/'
df = pd.read_excel(file_path+"CreditCard_Defaults.xlsx")

# Pre-processing
Marital = df['Marital_Status']
Marital.head()
df =df.drop(['Customer', 'Marital_Status'], axis=1)
Marital = pd.DataFrame(Marital.astype(int))
df1 = pd.concat([df, Marital], axis=1)

df1.dtypes

# Create an attribute map.
attribute_map = {
    'Default':[1,(0,1),[0,0]],
    'Gender':[1,(1, 2),[0,0]],
    'Education':[2,(0,1,2,3,4,5,6),[0,0]], 
    'Marital_Status':[2,(0,1,2,3), [0,0]],
    'card_class':[2,(1,2,3),[0,0]],
    'Age':[0,(20,80),[0,0]],
    'Credit_Limit':[0,(100, 80000),[0,0]],
    'Jun_Status':[0,(-2, 8),[0,0]],
    'May_Status':[0,(-2, 8),[0,0]],
    'Apr_Status':[0,(-2, 8),[0,0]],
    'Mar_Status':[0,(-2, 8),[0,0]],
    'Feb_Status':[0,(-2, 8),[0,0]],
    'Jan_Status':[0,(-2, 8),[0,0]],
    'Jun_Bill':[0,(-12000, 32000),[0,0]],
    'May_Bill':[0,(-12000, 32000),[0,0]],
    'Apr_Bill':[0,(-12000, 32000),[0,0]],
    'Mar_Bill':[0,(-12000, 32000),[0,0]],
    'Feb_Bill':[0,(-12000, 32000),[0,0]],
    'Jan_Bill':[0,(-12000, 32000),[0,0]],
    'Jun_Payment':[0,(0, 60000),[0,0]],
    'May_Payment':[0,(0, 60000),[0,0]],
    'Apr_Payment':[0,(0, 60000),[0,0]],
    'Mar_Payment':[0,(0, 60000),[0,0]],
    'Feb_Payment':[0,(0, 60000),[0,0]],
    'Jan_Payment':[0,(0, 60000),[0,0]],
    'Jun_PayPercent':[0,(0, 1),[0,0]],
    'May_PayPercent':[0,(0, 1),[0,0]],
    'Apr_PayPercent':[0,(0, 1),[0,0]],
    'Mar_PayPercent':[0,(0, 1),[0,0]],
    'Feb_PayPercent':[0,(0, 1),[0,0]],
    'Jan_PayPercent':[0,(0, 1),[0,0]]
}

# Use replace impute function to encode data
rie = ReplaceImputeEncode(data_map=attribute_map, drop=False, interval_scale='std', nominal_encoding='one-hot', display=True)
encoded_df = rie.fit_transform(df1)

len(encoded_df.dtypes) # 42 columns

# Create X dataframe and target variabel dataframe
X = encoded_df.drop(['Default'], axis=1)
Y = encoded_df['Default']
np_y = np.ravel(Y)

# Run neural network model

print("\n ***** Run Neural network ***** ")
neurons = [1, 2, 3, 5, 11, (2, 1), (2, 2), (5, 4), (6, 5), (7, 6)]
for n in neurons:
    
    print("\n ***** Run Neural network", str(n),"***** ")
    
    fnn = MLPClassifier(hidden_layer_sizes=n, activation='logistic', solver='lbfgs', max_iter=1000, random_state=12345)

    fnn = fnn.fit(X, np_y)
    NeuralNetwork.display_binary_metrics(fnn, X, np_y)

    # Neural Network Cross-Validation
    score_list = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']
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

# Split data into train-test and predict on test data
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state = 100)

fnn = MLPClassifier(hidden_layer_sizes=(2,2), activation='logistic', solver='lbfgs', max_iter=1000, random_state=12345)

np_train_y = np.ravel(train_y)
np_test_y = np.ravel(test_y)

fnn = fnn.fit(train_x, np_train_y)
cf = NeuralNetwork.display_binary_metrics(fnn, test_x, np_test_y)

pred_test = fnn.predict(test_x)

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(test_y, pred_test)


tn = conf_mat[0,0]
fp = conf_mat[0,1]
fn = conf_mat[1,0]
tp = conf_mat[1,1]

accuracy=(tp+tn)/(tp+tn+fp+fn)
recall=(tp)/(tp+fn)
precision=(tp)/(tp+fp)
f1=(2*recall*precision)/(recall+precision)


print("accuracy: %f" %accuracy) #0.87166
print("recall: %f" %recall) #0.441645
print("precision: %f" %precision) #0.68028
print("F1 score: %f" %f1) #0.535585

print("***********************************************************")
print("END OF NEURAL NETWORK MODEL \n")
print("***********************************************************")


print("\n\n")

print("***********************************************************")
print("RANDOM FOREST MODEL \n")
print("***********************************************************")


# coding: utf-8


## Random Forest

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score
from pydotplus import graph_from_dot_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,confusion_matrix

## Change path 

import sys
sys.path.append("/Users/deepthisen/Desktop/Courses/stat656/MidTerm Exam/")

from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score



from Class_tree import DecisionTree
from Class_replace_impute_encode import ReplaceImputeEncode




## Change Path 
file_path = '/Users/deepthisen/Desktop/Courses/stat656/MidTerm Exam/'
df = pd.read_excel(file_path+"CreditCard_Defaults.xlsx")




attribute_map = {
    'Default':[1,(0,1),[0,0]],
    'Gender':[1,(1, 2),[0,0]],
    'Education':[2,(0,1,2,3,4,5,6),[0,0]], 
    'Marital_Status':[2,(0,1,2,3), [0,0]],
    'card_class':[2,(1,2,3),[0,0]],
    'Age':[0,(20,80),[0,0]],
    'Credit_Limit':[0,(100, 80000),[0,0]],
    'Jun_Status':[0,(-2,8),[0,0]],
    'May_Status':[0,(-2,8),[0,0]],
    'Apr_Status':[0,(-2,8),[0,0]],
    'Mar_Status':[0,(-2,8),[0,0]],
    'Feb_Status':[0,(-2,8),[0,0]],
    'Jan_Status':[0,(-2,8),[0,0]],
    'Jun_Bill':[0,(-12000, 32000),[0,0]],
    'May_Bill':[0,(-12000, 32000),[0,0]],
    'Apr_Bill':[0,(-12000, 32000),[0,0]],
    'Mar_Bill':[0,(-12000, 32000),[0,0]],
    'Feb_Bill':[0,(-12000, 32000),[0,0]],
    'Jan_Bill':[0,(-12000, 32000),[0,0]],
    'Jun_Payment':[0,(0, 60000),[0,0]],
    'May_Payment':[0,(0, 60000),[0,0]],
    'Apr_Payment':[0,(0, 60000),[0,0]],
    'Mar_Payment':[0,(0, 60000),[0,0]],
    'Feb_Payment':[0,(0, 60000),[0,0]],
    'Jan_Payment':[0,(0, 60000),[0,0]],
    'Jun_PayPercent':[0,(0, 1),[0,0]],
    'May_PayPercent':[0,(0, 1),[0,0]],
    'Apr_PayPercent':[0,(0, 1),[0,0]],
    'Mar_PayPercent':[0,(0, 1),[0,0]],
    'Feb_PayPercent':[0,(0, 1),[0,0]],
    'Jan_PayPercent':[0,(0, 1),[0,0]]
}




df.drop(['Customer'], axis=1)



rie = ReplaceImputeEncode(data_map=attribute_map, drop=False, nominal_encoding='one-hot', display=True)
encoded_df = rie.fit_transform(df)


## Splitting into Target and Predictors

varlist = df['Default']
X = encoded_df.drop('Default', axis=1)
y = encoded_df['Default']
np_y = np.ravel(y)



## Cross Validation RANDOM FOREST 

from sklearn.ensemble import RandomForestClassifier

estimators_list   = [20,30,50,70,90]
max_features_list = ['auto','sqrt',0.3, 0.5, 0.7]
score_list = ['accuracy', 'recall', 'precision', 'f1']
max_f1 = 0
max_AUC = 0
mean_list = []
std_list = []

for e in estimators_list:
    for f in max_features_list:
        print("\nNumber of Trees: ", e, " Max_features: ", f)
        rfc = RandomForestClassifier(n_estimators=e, criterion="gini", max_depth=None, min_samples_split=20,min_samples_leaf=1, max_features=f,\
		                             n_jobs=-1, bootstrap=True, random_state=12345)
        rfc= rfc.fit(X, np_y)
        scores = cross_validate(rfc, X, np_y, scoring=score_list,return_train_score=False, cv=10)
        
        print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
        for s in score_list:
            var = "test_"+s
            mean = scores[var].mean()
            std  = scores[var].std()
            mean_list.append(mean)
            std_list.append(std)
            print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
        if mean > max_AUC:
            max_AUC = mean
            best_estimator    = e
            best_max_features = f

print("\nBest based on F1-Score")
print("Best Number of Estimators (trees) = ", best_estimator)
print("Best Maximum Features = ", best_max_features)


## Splitting and fitting the model
X_train, X_validate, y_train, y_validate = train_test_split(X, np_y,test_size = 0.3, random_state=12345)

rfc = RandomForestClassifier(n_estimators=50, criterion="gini", max_depth=None, min_samples_split=20,min_samples_leaf=1, max_features=0.7, \
                               n_jobs=1, bootstrap=True, random_state=12345)
rfc= rfc.fit(X_train, y_train)



# Printing Test Results
print("*****RESULTS *******")
DecisionTree.display_binary_split_metrics(rfc, X_train, y_train,X_validate, y_validate)
features = X_train.columns.values.tolist()
DecisionTree.display_importance(rfc, features)
print("*****END OF RESULTS *******")



# Code to find the best threshold under a given set of conditions
# Enhancements for threshold adjustment for better sensitivity

print("*****SENSITIVITY ENHANCEMENT FOR RANDOM FORESTS *******")
threshold_array = np.linspace(0.15,0.85,num =140)
best_metric = 0
min_sensitivity = 0.7
min_accuracy = 0.8 
i = 0

best_threshold_acc = "infeasible under given conditions"
best_threshold_sen = "infeasible under given conditions"
best_threshold = "infeasible under given conditions"
predict_probabilities = rfc.predict_proba(X_validate)

for i in threshold_array:
    
    predict_mine = np.where(predict_probabilities < i,0,1)
    k = confusion_matrix(y_validate, predict_mine[:,1])

    True_Predictions = k[0,0]+k[1,1]
    False_pos = k[1,0]
    True_pos = k[1,1]
    True_neg = k[0,0]
    False_neg = k[0,1]
    Total = True_Predictions + False_pos + False_neg
    Accuracy =(True_Predictions/Total)
    Sensitivity =(True_pos)/(k[1,0]+k[1,1])
    
    if (Accuracy + Sensitivity) > best_metric and Accuracy > min_accuracy and Sensitivity > min_sensitivity:
        best_metric = (Accuracy + Sensitivity)
        best_threshold_acc = Accuracy
        best_threshold_sen = Sensitivity
        best_threshold = i

print("\nBest Model")
print("Accuracy....",best_threshold_acc)
print("Sensitivity..",best_threshold_sen)
print("Threshold....",best_threshold)

print("***********************************************************")
print("END OF RANDOM FOREST MODEL \n")
print("***********************************************************")
