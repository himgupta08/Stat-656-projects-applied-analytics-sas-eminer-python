
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
DecisionTree.display_binary_split_metrics(rfc, X_train, y_train,X_validate, y_validate)
features = X_train.columns.values.tolist()
DecisionTree.display_importance(rfc, features)




# Code to find the best threshold under a given set of conditions
# Enhancements for threshold adjustment for better sensitivity
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

