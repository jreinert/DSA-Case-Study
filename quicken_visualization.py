# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:16:16 2021

@author: reine
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

#set pandas columns/rows to show all
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#import data
df = pd.read_csv('DSA Dataset Cleaned.csv', encoding='iso-8859-1')
df1 = pd.read_csv('DSA Data Set.csv', encoding='iso-8859-1')
df = df.drop(columns=['Unnamed: 0']) #drop unnecessary column added when written to csv

#pairplots - numerical data
sns.pairplot(df, hue='Subscribed to Term Deposit')

#encode categorical data into new dataframe
cat_features = ['Age Ranges', 'Job', 'Marital Status',
                'Education Level', 'Housing Loan', 'Personal Loan',
                'Contact Method', 'Last Contact Month', 'Last Contact Day',
                'Prior or New Contact', 'Outcome of Previous Campaign', 
                'Probability Category', 'Subscribed to Term Deposit']
df2 = df.copy(deep=True)
label_enc = LabelEncoder()
for x in cat_features:
    df2[x] = label_enc.fit_transform(df2[x].values)

df1['y'] = label_enc.fit_transform(df1['y'].values)
df1['ModelPrediction'] = label_enc.fit_transform(df1['ModelPrediction'].values)

#correlation matrix visualization
corr_matrix = df2.corr()
sns.set(font_scale=.6)
plt.figure(figsize=(17, 12))
sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), cmap='YlOrRd', square=True, annot=True, cbar=False)

#correlation matrix visualization for ModelPrediction and y with pre-cleaned data
df3 = df1[['ModelPrediction', 'y']].copy()
corr_matrix = df3.corr()
sns.set(font_scale=.75)
plt.figure()
sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), cmap="YlGnBu", square=True, annot=True, cbar=False)

#age_bin barplot
sns.set(style='whitegrid')
plt.figure()
bar = sns.countplot(x='Age Ranges', hue='Subscribed to Term Deposit', order=['<32', '32-37', '38-46', '>47'], data=df)
plt.show()

#month barplot
sns.set(style='whitegrid')
plt.figure()
bar = sns.countplot(x='Last Contact Month', hue='Subscribed to Term Deposit', order=['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sept', 'oct', 'nov', 'dec'], data=df)
plt.show()

#prob_bin barplot
sns.set(style='whitegrid')
plt.figure()
bar = sns.countplot(x='Probability Category', hue='Subscribed to Term Deposit', order=['<50th Percentile', '>=50th Percentile'], data=df)
plt.show()

#remaining categorical feature barplots:
for cat in cat_features:
    if cat == 'Age_Ranges' or cat == 'Subscribed to Term Deposit' or cat == 'Last Contact Month' or cat == 'Probability Category':
        continue;
    else:
        sns.set(style='whitegrid')
        plt.figure()
        bar = sns.countplot(x=cat, hue='Subscribed to Term Deposit', data=df)
        plt.show()

num_features = ['Age', 'Duration of Last Contact', 'Current Campaign Contacts', 
                'Days Since Contact From Last Campaign', 'Number of Prior Contacts', 
                'Employee Variation Rate', 'Consumer Price Index', 'Euribor 3 Month Rate', 
                'Number of Employees', 'ModelPrediction']
for feat in num_features:
    sns.set_theme(style='whitegrid')
#   plt.figure()
    hist = sns.displot(x=feat, hue='Subscribed to Term Deposit', data=df, multiple='dodge')
    plt.show()
    