# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 12:37:24 2021

@author: reine
"""
def key_performance_indicators(confusion_matrix):
    cm = confusion_matrix
    print(cm)
    
    # Total accuracy and misclassification calculations
    total_right = cm[0][0] + cm[1][1]
    total_wrong = cm[0][1] + cm[1][0]
    total_acc = total_right / (total_right + total_wrong)
    print(f'The total accuracy of the Model is {total_acc:.3f}')
    
    misclass = total_wrong / (total_right + total_wrong)
    print(f'The total misclassification rate (error) of the Model is {misclass:.3f}')
    
    # TP, FP, FN - Precision, Recall, Specificity, F1 calculations
    tp = cm[0][0]
    fp = cm[1][0]
    fn = cm[0][1]
    tn = cm[1][1]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = (2 * precision * recall) / (precision + recall)
    
    # Print KPI
    print(f'The precision of the Model is {precision:.3f}')
    print(f'The recall of the Model is {recall:.3f}')
    print(f'The specificity of the Model is {specificity:.3f}')
    print(f'The F1 Score of the Model is {f1:.3f}')

#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense

#import the dataset
df = pd.read_csv('DSA Dataset Cleaned.csv', encoding='iso-8859-1')
df = df.drop(columns=['Unnamed: 0']) #drop unnecessary column added when written to csv

#drop features not being used in the model
df = df.drop(columns=['Age', 'Age Ranges', 'Job', 'Marital Status',
                      'Education Level', 'Housing Loan', 'Personal Loan',
                      'Last Contact Month', 'Last Contact Day', 'Contact Method',
                      'Duration of Last Contact', 'Current Campaign Contacts', 'Consumer Price Index',
                      'Days Since Contact From Last Campaign', 'Consumer Confidence Index',
                      'ModelPrediction', 'Probability Category'])

# One Hot Encoding
df = pd.get_dummies(df, drop_first = True)

# Split into X/Y and Train/Test
X = df.iloc[:, 0:-1] 
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Build the Model
input_dims = len(X.columns)
num_neurons = 14
total_num_hidden_layers = 4
num_epochs = 50
num_batchs = 15
act_func = 'relu'

model = Sequential()
model.add(Dense(num_neurons, input_dim=input_dims, activation=act_func))

i = 0
while i < total_num_hidden_layers-1:
    model.add(Dense(units = num_neurons, kernel_initializer = 'uniform', activation = act_func))
    i += 1

model.add(Dense(1, activation='sigmoid'))

#Train the Model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batchs, verbose=1)

#Predict the Purchase Amounts
y_predict = model.predict(X_test)
r_squared = r2_score(y_test, y_predict)
print('r squared: ', r_squared)

y_predict = (y_predict > 0.5)

# Evaluate the Model
cm = confusion_matrix(y_test, y_predict)
key_performance_indicators(cm)

sns.set(font_scale=.75)
plt.figure()
sns.heatmap(cm, mask=np.zeros_like(cm, dtype=np.bool), cmap="YlGnBu", square=True, annot=True, cbar=False, fmt='d', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
