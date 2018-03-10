#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 00:56:12 2018

@author: eJones
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report 
from sklearn.metrics import roc_curve, auc
import numpy as np
from math import sqrt

class DecisionTree(object):
    
    def display_metrics(dt, X, y):
        predictions = dt.predict(X)
        print("\nModel Metrics")
        print("{:.<23s}{:15d}".format('Observations', X.shape[0]))
        print("{:.<23s}{:15d}".format('Coefficients', X.shape[1]+1))
        print("{:.<23s}{:15d}".format('DF Error', X.shape[0]-X.shape[1]-1))
        R2 = r2_score(y, predictions)
        print("{:.<23s}{:15.4f}".format('R-Squared', R2))
        print("{:.<23s}{:15.4f}".format('Mean Absolute Error', \
                      mean_absolute_error(y,predictions)))
        print("{:.<23s}{:15.4f}".format('Median Absolute Error', \
                      median_absolute_error(y,predictions)))
        print("{:.<23s}{:15.4f}".format('Avg Squared Error', \
                      mean_squared_error(y,predictions)))
        print("{:.<23s}{:15.4f}".format('Square Root ASE', \
                      sqrt(mean_squared_error(y,predictions))))
        
    def display_importance(dt, col):
        nx = dt.n_features_
        max_label = 6
        for i in range(len(col)):
            if len(col[i]) > max_label:
                max_label = len(col[i])+4
        label_format = ("{:.<%i" %max_label)+"s}{:9.4f}"
        
        features = []
        this_col = []
        for i in range(nx):
            features.append(dt.feature_importances_[i])
            this_col.append(col[i])
        sorted = False
        while (sorted==False):
            sorted = True
            for i in range(nx-1):
                if features[i]<features[i+1]:
                    sorted=False
                    x = features[i]
                    c = this_col[i]
                    features[i] = features[i+1]
                    this_col[i] = this_col[i+1]
                    features[i+1] = x
                    this_col[i+1] = c
        print("")
        label_format2 = ("{:.<%i" %max_label)+"s}{:s}"
        print(label_format2.format("FEATURE", " IMPORTANCE"))
        for i in range(nx):
            print(label_format.format(this_col[i], features[i]))
        print("")
        
    def display_binary_metrics(dt, X, y):
        if len(dt.classes_) != 2:
            print("****Error - this target does not appear to be binary.")
            print("****Try using logreg.nominal_binary_metrics() instead.")
            return
        numpy_y = np.array(y)
        z = np.zeros(len(y))
        predictions = dt.predict(X) # get binary class predictions
        conf_mat = confusion_matrix(y_true=y, y_pred=predictions)
        misc = 100*(conf_mat[0][1]+conf_mat[1][0])/(len(y))
        for i in range(len(y)):
            if numpy_y[i] == 1:
                z[i] = 1
        probability = dt.predict_proba(X) # get binary probabilities
        #probability = dt.predict_proba(X)
        print("\nModel Metrics")
        print("{:.<27s}{:10d}".format('Observations', X.shape[0]))
        print("{:.<27s}{:10d}".format('Features', X.shape[1]))
        print("{:.<27s}{:10d}".format('Maximum Tree Depth',\
                              dt.max_depth))
        print("{:.<27s}{:10d}".format('Minimum Leaf Size', \
                              dt.min_samples_leaf))
        print("{:.<27s}{:10d}".format('Minimum split Size', \
                              dt.min_samples_split))
        print("{:.<27s}{:10.4f}".format('Mean Absolute Error', \
                      mean_absolute_error(z,probability[:, 1])))
        print("{:.<27s}{:10.4f}".format('Avg Squared Error', \
                      mean_squared_error(z,probability[:, 1])))
        acc = accuracy_score(y, predictions)
        print("{:.<27s}{:10.4f}".format('Accuracy', acc))
        pre = precision_score(y, predictions)
        print("{:.<27s}{:10.4f}".format('Precision', pre))
        tpr = recall_score(y, predictions)
        print("{:.<27s}{:10.4f}".format('Recall (Sensitivity)', tpr))
        f1 =  f1_score(y,predictions)
        print("{:.<27s}{:10.4f}".format('F1-Score', f1))
        print("{:.<27s}{:9.1f}{:s}".format(\
                'MISC (Misclassification)', misc, '%'))
        n_    = [conf_mat[0][0]+conf_mat[0][1], conf_mat[1][0]+conf_mat[1][1]]
        miscc = [100*conf_mat[0][1]/n_[0], 100*conf_mat[1][0]/n_[1]]
        for i in range(2):
            print("{:s}{:.<16.0f}{:>9.1f}{:<1s}".format(\
                  '     class ', dt.classes_[i], miscc[i], '%'))      

        print("\n\n     Confusion")
        print("       Matrix    ", end="")
        for i in range(2):
            print("{:>7s}{:<3.0f}".format('Class ', dt.classes_[i]), end="")
        print("")
        for i in range(2):
            print("{:s}{:.<6.0f}".format('Class ', dt.classes_[i]), end="")
            for j in range(2):
                print("{:>10d}".format(conf_mat[i][j]), end="")
            print("")
        print("")
        
    def display_binary_split_metrics(dt, Xt, yt, Xv, yv):
        if len(dt.classes_) != 2:
            print("****Error - this target does not appear to be binary.")
            print("****Try using logreg.nominal_binary_metrics() instead.")
            return
        numpy_yt = np.array(yt)
        numpy_yv = np.array(yv)
        zt = np.zeros(len(yt))
        zv = np.zeros(len(yv))
        #zt = deepcopy(yt)
        for i in range(len(yt)):
            if numpy_yt[i] == 1:
                zt[i] = 1
        for i in range(len(yv)):
            if numpy_yv[i] == 1:
                zv[i] = 1
        predict_t = dt.predict(Xt)
        predict_v = dt.predict(Xv)
        conf_matt = confusion_matrix(y_true=yt, y_pred=predict_t)
        conf_matv = confusion_matrix(y_true=yv, y_pred=predict_v)
        prob_t = dt.predict_proba(Xt)
        prob_v = dt.predict_proba(Xv)
        print("\n")
        print("{:.<23s}{:>15s}{:>15s}".format('Model Metrics', \
                                      'Training', 'Validation'))
        print("{:.<23s}{:15d}{:15d}".format('Observations', \
                                          Xt.shape[0], Xv.shape[0]))
        
        print("{:.<23s}{:15d}{:15d}".format('Features', Xt.shape[1], \
                                                          Xv.shape[1]))
        print("{:.<23s}{:15d}{:15d}".format('Minimum Leaf Size', \
                              dt.min_samples_leaf, dt.min_samples_leaf))
        print("{:.<23s}{:15d}{:15d}".format('Minimum split Size', \
                              dt.min_samples_split, dt.min_samples_split))
        
        print("{:.<23s}{:15.4f}{:15.4f}".format('Mean Absolute Error', \
                      mean_absolute_error(zt,prob_t[:,1]), \
                      mean_absolute_error(zv,prob_v[:,1])))
        print("{:.<23s}{:15.4f}{:15.4f}".format('Avg Squared Error', \
                      mean_squared_error(zt,prob_t[:,1]), \
                      mean_squared_error(zv,prob_v[:,1])))
        
        acct = accuracy_score(yt, predict_t)
        accv = accuracy_score(yv, predict_v)
        print("{:.<23s}{:15.4f}{:15.4f}".format('Accuracy', acct, accv))
        
        print("{:.<23s}{:15.4f}{:15.4f}".format('Precision', \
                      precision_score(yt,predict_t), \
                      precision_score(yv,predict_v)))
        print("{:.<23s}{:15.4f}{:15.4f}".format('Recall (Sensitivity)', \
                      recall_score(yt,predict_t), \
                      recall_score(yv,predict_v)))
        print("{:.<23s}{:15.4f}{:15.4f}".format('F1-score', \
                      f1_score(yt,predict_t), \
                      f1_score(yv,predict_v)))
        misct = conf_matt[0][1]+conf_matt[1][0]
        miscv = conf_matv[0][1]+conf_matv[1][0]
        misct = 100*misct/len(yt)
        miscv = 100*miscv/len(yv)
        n_t   = [conf_matt[0][0]+conf_matt[0][1], \
                 conf_matt[1][0]+conf_matt[1][1]]
        n_v   = [conf_matv[0][0]+conf_matv[0][1], \
                 conf_matv[1][0]+conf_matv[1][1]]
        misc_ = [[0,0], [0,0]]
        misc_[0][0] = 100*conf_matt[0][1]/n_t[0]
        misc_[0][1] = 100*conf_matt[1][0]/n_t[1]
        misc_[1][0] = 100*conf_matv[0][1]/n_v[0]
        misc_[1][1] = 100*conf_matv[1][0]/n_v[1]
        print("{:.<27s}{:10.1f}{:s}{:14.1f}{:s}".format(\
                'MISC (Misclassification)', misct, '%', miscv, '%'))
        for i in range(2):
            print("{:s}{:.<16.0f}{:>10.1f}{:<1s}{:>14.1f}{:<1s}".format(\
                  '     class ', dt.classes_[i], \
                  misc_[0][i], '%', misc_[1][i], '%'))
        print("\n\nTraining")
        print("Confusion Matrix ", end="")
        for i in range(2):
            print("{:>7s}{:<3.0f}".format('Class ', dt.classes_[i]), end="")
        print("")
        for i in range(2):
            print("{:s}{:.<6.0f}".format('Class ', dt.classes_[i]), end="")
            for j in range(2):
                print("{:>10d}".format(conf_matt[i][j]), end="")
            print("")
        # In the binary case, the classification report is incorrect
        #cr = classification_report(yv, predict_v, dt.classes_)
        #print("\n",cr)
        
        print("\n\nValidation")
        print("Confusion Matrix ", end="")
        for i in range(2):
            print("{:>7s}{:<3.0f}".format('Class ', dt.classes_[i]), end="")
        print("")
        for i in range(2):
            print("{:s}{:.<6.0f}".format('Class ', dt.classes_[i]), end="")
            for j in range(2):
                print("{:>10d}".format(conf_matv[i][j]), end="")
            print("")
        # In the binary case, the classification report is incorrect
        #cr = classification_report(yv, predict_v, dt.classes_)
        #print("\n",cr)
   