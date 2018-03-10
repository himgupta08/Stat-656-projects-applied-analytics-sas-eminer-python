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

