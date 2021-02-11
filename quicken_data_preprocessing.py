# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 20:13:17 2021

@author: reine
"""

#import libraries
import pandas as pd
import numpy
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

#custom functions
def categoricalValueCounts(df, cat_features: list):
    print('\n*** Value counts for categorical features ***')
    for cat in cat_features:
        print(f'\n --- Feature: {cat} ---')
        vals = df[cat].value_counts()
        print(f'{vals}')
        
        try:
            num_unknown = vals['unknown'] #num of 'unknown' values
            total_size = df[cat].size #total num of values
            percent_unknown = (num_unknown / total_size) * 100
            print(f'{percent_unknown:.2f}% unknown values')    
        except:
            continue    

def calcYesNoForY(df):
    y_vals = df['y'].value_counts()
    y_yes_percent = (y_vals['yes'] / y_vals.sum()) * 100
    y_no_percent = (y_vals['no'] / y_vals.sum()) * 100
    print(f'The percent of yes y values is: {y_yes_percent:.2f}%')
    print(f'The percent of no y values is: {y_no_percent:.2f}%')

def exploreModelPredictionData(df, percentile: float):
    prob_pred_group_a = df.groupby('y')['ModelPrediction'].apply(lambda x:x[x < df['ModelPrediction'].quantile(percentile)].count())
    print(f'\ny value counts where ModelPrediction < {percentile*100}th percentile')
    print(prob_pred_group_a)
    
    prob_pred_group_b = df.groupby('y')['ModelPrediction'].apply(lambda x:x[x >= df['ModelPrediction'].quantile(percentile)].count())
    print(f'\ny value counts where ModelPrediction >= {percentile*100}th percentile')
    print(prob_pred_group_b)

#import data
df = pd.read_csv('DSA Data Set.csv', encoding='iso-8859-1')

#set pandas columns/rows to show all
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


#get descriptive statistics on numerical features
print('--- Analysis of initial dataset ---')
print('*** Descriptive statistics on numerical features ***')
print(df.describe())

#get value counts for categorical features
cat_features = ['job','marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y'] #list of categorical features for looping
categoricalValueCounts(df, cat_features)

#calculate y/n percents for y
calcYesNoForY(df)

#explore ModelPrediction accuracy
exploreModelPredictionData(df, .5)
exploreModelPredictionData(df, .25)

print('\n *** Explore 999 Data ****')
#get # of 999 values for pdays column
pdays_series = pd.Series(df['pdays']) #create series of pdays values
num_nines = pdays_series[pdays_series == 999].count() #count number of 999 values
total_series_size = pdays_series.size #total num of values
percent_nines = (num_nines / total_series_size) * 100
print(f'\nThe total number of 999 values: {num_nines}')
print(f'The percent of 999 values: {percent_nines:.2f}%')

#groupby pdays, poutcome and count previous values
prev_camp_group = df.groupby(['pdays', 'poutcome'])['previous'].value_counts()
print(f'\n{prev_camp_group}')
sum_of_nines_failure = prev_camp_group[999]['failure'].sum()
sum_of_nines_nonexistent = prev_camp_group[999]['nonexistent'].sum()
total_sum_of_nines = sum_of_nines_failure + sum_of_nines_nonexistent
percent_nines_failure = (sum_of_nines_failure / total_sum_of_nines) * 100
percent_nines_nonexistent = (sum_of_nines_nonexistent / total_sum_of_nines) * 100
print(f'The total number 999 pdays rows where poutcomes is failure is: {sum_of_nines_failure:,} and is {percent_nines_failure:.2f}% of the total')
print(f'The total number 999 pdays rows where poutcomes is nonexistent is: {sum_of_nines_nonexistent:,} and is {percent_nines_nonexistent:.2f}% of the total')

#create new dataframe to drop columns & rows without altering the original dataframe
df2 = df
df2 = df2.drop(columns='default') #drop default column
cat_features.remove('default') #remove default from cat_features list

#loop through cat_features and remove 'unknown' rows from df2
for cat in cat_features:
    df2 = df2[df2[cat] != 'unknown']

#drop rows where pdays is showing no contact but poutcome is showing as a failure and previous is showing contacts    
df2 = df2.drop(df2[(df2['pdays'] == 999) & (df2['poutcome'] == 'failure')].index)

#rewrite entrepreneur to self-employed, housemaid to services, admin. & management to white-collar in jobs feature
print('\n Rewriting job values...')
for i in df2.index:
    if df2.at[i,'job'] == 'entrepreneur':
        df2.at[i,'job'] = 'self-employed'
    elif df2.at[i,'job'] == 'housemaid':
        df2.at[i,'job'] = 'services'
    elif df2.at[i,'job'] == 'admin.' or df2.at[i,'job'] == 'management':
        df2.at[i,'job'] = 'white-collar'

#rewrite basic.4y, basic.6y, basic.9y to basic, high.school to high school, professional.course to professional course, and university.degree to university degree
print('\n Rewriting education values...')
for i in df2.index:
    if df2.at[i,'education'] == 'basic.4y' or df2.at[i,'education'] == 'basic.6y' or df2.at[i,'education'] == 'basic.9y':
        df2.at[i,'education'] = 'basic <=9y'
    elif df2.at[i,'education'] == 'high.school':
        df2.at[i,'education'] = 'high school'
    elif df2.at[i,'education'] == 'professional.course':
        df2.at[i,'education'] = 'professional course'
    elif df2.at[i,'education'] == 'university.degree':
        df2.at[i,'education'] = 'university degree'

#add column to dataframe re: prior contact/no prior contact and populate with null values
df2.insert(loc=13, column='prior_new_contact', value=['' for i in range(df2.shape[0])])

#rewrite null values in column created above to either new contact or prior contact
print('\n New column added to dataframe (prior_new_contact) and values being written...')
for i in df2.index:
    if df2.at[i, 'pdays'] == 999:
        df2.at[i,'prior_new_contact'] = 'new contact'
    else:
        df2.at[i,'prior_new_contact'] = 'prior contact'
        
#add column to dataframe re: age bins
df2.insert(loc=1, column='age_bin', value=['' for i in range(df2.shape[0])])

#rewrite null values in column created above to appropriate age bin
print('\n New column added to dataframe (age_bin) and values being written...')
for i in df2.index:
    if df2.at[i, 'age'] < df2['age'].quantile(.25):
        df2.at[i,'age_bin'] = f"""<{int(df2['age'].quantile(.25))}"""
    elif df2.at[i, 'age'] >= df2['age'].quantile(.25) and df2.at[i, 'age'] < df2['age'].quantile(.5):
        df2.at[i,'age_bin'] = f"""{int(df2['age'].quantile(.25))}-{int(df2['age'].quantile(.5)-1)}"""
    elif df2.at[i, 'age'] >= df2['age'].quantile(.5) and df2.at[i, 'age'] < df2['age'].quantile(.75):
        df2.at[i,'age_bin'] = f"""{int(df2['age'].quantile(.5))}-{int(df2['age'].quantile(.75)-1)}"""
    elif df2.at[i, 'age'] >= df2['age'].quantile(.75):
        df2.at[i,'age_bin'] = f""">{int(df2['age'].quantile(.75))}"""

#add column to dataframe re: ModelPrediction prob bins
df2.insert(loc=22, column='prob_bin', value=['' for i in range(df2.shape[0])])

#rewrite null values in column created above to appropriate  model_prob
print('\n New column added to dataframe (prob_bin) and values being written...')
for i in df2.index:
    if df2.at[i, 'ModelPrediction'] < df2['ModelPrediction'].quantile(.5):
        df2.at[i,'prob_bin'] = '<50th Percentile'
    elif df2.at[i, 'ModelPrediction'] >= df2['ModelPrediction'].quantile(.5):
        df2.at[i,'prob_bin'] = '>=50th Percentile'

#explore data for df2
print('\n --- Analysis of cleaned dataset ---')
print('*** Descriptive statistics on numerical features ***')
print(df2.describe())
cat_features.append('age_bin')
cat_features.append('prob_bin')
categoricalValueCounts(df2, cat_features)
calcYesNoForY(df2)
exploreModelPredictionData(df2, .5)
exploreModelPredictionData(df2, .25)

#rename columns
df2 = df2.rename(columns={'age':'Age', 'age_bin':'Age Ranges', 'job': 'Job',
                    'marital': 'Marital Status', 'education': 'Education Level',
                    'housing': 'Housing Loan', 'loan': 'Personal Loan',
                    'contact': 'Contact Method', 'month': 'Last Contact Month', 
                    'day_of_week': 'Last Contact Day','duration': 'Duration of Last Contact', 
                    'campaign': 'Current Campaign Contacts', 'pdays': 'Days Since Contact From Last Campaign',
                    'previous': 'Number of Prior Contacts', 'prior_new_contact': 'Prior or New Contact',
                    'poutcome': 'Outcome of Previous Campaign', 'emp.var.rate': 'Employee Variation Rate',
                    'cons.price.idx': 'Consumer Price Index', 'cons.conf.idx': 'Consumer Confidence Index',
                    'euribor3m': 'Euribor 3 Month Rate', 'nr.employed': 'Number of Employees', 'prob_bin':'Probability Category',
                    'y': 'Subscribed to Term Deposit'})


#write clean dataset to csv
print('\nCleaned dataframe being written to new csv file')
df2.to_csv("DSA Dataset Cleaned.csv")
     

                 