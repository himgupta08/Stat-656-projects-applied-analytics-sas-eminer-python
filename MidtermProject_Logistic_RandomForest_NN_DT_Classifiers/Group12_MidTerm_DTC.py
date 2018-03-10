#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 01:57:47 2018

@author: deepthisen
"""

from Class_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
#from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from pydotplus import graph_from_dot_data
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from Class_replace_impute_encode import ReplaceImputeEncode
import graphviz
import pandas as pd
import numpy as np
#from matplotlib import pyplot as pt

file_path = '/Users/deepthisen/Desktop/Courses/stat656/MidTerm Exam/'
df = pd.read_excel(file_path+"CreditCard_Defaults.xlsx")
df=df.drop(['Customer'],axis=1)
print("Authentication Data with %i observations & %i attributes.\n" %df.shape, df[0:2])

attribute_map = {
    'Gender':[1,(1,2),[0,0]], 
    'Education':[2,(0,1,2,3,4,5,6),[0,0]],
    'Marital_Status' :[2,(0,1,2,3),[0,0]],         
    'card_class':[2,(1,2,3),[0,0]],       
    'Age'  :[0,(20,80),[0,0]], #
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
# Now instantiate the class passing in the parameters setting your choices
rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding=encoding, \
                          interval_scale = scale, drop=False, display=True)
# Now request replace-impute-encode for your dataframe

encoded_df = rie.fit_transform(df)
varlist = ['Default']
X = encoded_df
y = df[varlist]

maxdep_list=[4,5,6,7,8,9]
min_samp_leaf_list=[3,4,5,6]
min_samp_split_list=[3,4,5,6]
score_list = ['accuracy', 'recall', 'precision', 'f1']
bestf1mean=0
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
                    #Choosing the depth, min_leaf and min_split with the best f1
                    """
                    if s=='f1':
                        if mean>bestf1mean:
                            bestf1mean=mean
                            bestdepth=maxdep
                            bestminleaf=min_leaf
                            bestminsplit=min_split
                    """
bestdepth=7
bestminleaf=4
bestminsplit=4
print("Best Max Depth Setting: %s" %bestdepth)
print("Best Min Leaf Sample Setting: %s" %bestminleaf)
print("Best Min Split Sample Setting: %s" %bestminsplit)

print("Splitting the dataset and comparing Training and Validation:")
X_train, X_validate, y_train, y_validate = \
            train_test_split(X,y,test_size = 0.3, random_state=7)
#Fitting a tree with the best depth using the training data
dtc1 = DecisionTreeClassifier(criterion='entropy', max_depth=bestdepth, \
min_samples_split=bestminsplit, min_samples_leaf=bestminleaf)
dtc1 = dtc1.fit(X_train,y_train)
features = list(encoded_df)
classes = ['No Default', 'Default']
DecisionTree.display_importance(dtc1, features)
DecisionTree.display_binary_metrics(dtc1, X_validate, y_validate)



print("Enhanced Sensitivity By Altering Probability Threshold")
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
metricarray=np.array(metriclist)
thresholdarray=np.linspace(0.0, 0.99, num=100)
#Plot if required
#pt.plot(thresholdarray[0:90],metricarray[0:90])

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
print("Confusion Matrix : Training")
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
print("Confusion Matrix : Validation")
print(k)
print("Validation Accuracy:  %s" %accu)
print("Validation Sensitivity:  %s" %sens)
print("Validation Specificity:  %s" %spec)

dot_data = export_graphviz(dtc1, filled=True, rounded=True, \
class_names=classes, feature_names = features, out_file=None)
#write tree to png file 'banknote_forged'
graph_png = graph_from_dot_data(dot_data)
graph_path = '/Users/deepthisen/Desktop/Courses/stat656/MidTerm Exam/graphs/'
graph_png.write_png(graph_path+'bestfitDTC.png')
#Display tree in pdf file 'banknote_forged.pdf'
graph_pdf = graphviz.Source(dot_data)
graph_pdf.view('bestfitDTC') #Displays tree
