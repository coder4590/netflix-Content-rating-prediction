import pandas as pd
import numpy as np
import csv
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding
from sklearn.preprocessing import StandardScaler , LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split,KFold, cross_val_score, RandomizedSearchCV
from scipy.stats import randint, uniform 
import xgboost as xgb

def cleanning_data():
    df=pd.read_csv('netflix_titles.csv')

    # the first step is the like of the check if the data has some of the missing values 
    print("Finding the Missing values")
    print(df.isnull().sum())

    # now we will remove the like of hte mssing value and handle then properly 
    # this row has the total of the 10 row missing like of the total of the 1 percent of the total data totally removable 
    df=df.dropna(subset=['date_added'])

    # this is contain 1 percent of the total of the missing value compared to the like of the original data
    df=df.dropna(subset=['rating'])

    # this also contian the like of the data missing data which is less than the like of the 1 percent 
    df=df.dropna(subset=['duration'])

    # this contain more than the like of the five percent so we will like of the fille them like with the na

    df['cast'].fillna('Unknown', inplace=True)     
    df['country'].fillna('Unknown', inplace=True)   
    df['director'].fillna('Unknown', inplace=True)

    # now the next thing is that the like of the remove the value which is like of the cause of the duplication and the like of it and the like of the which is to be removed by the such of the force which is still be unknown

    print(f"all of the duplicate row: {df.duplicated().sum()}")
    duplicates = df[df.duplicated(keep='first')]
    print(duplicates)

    df=df.drop_duplicates()
    # output:
    # there is out not a single like of the which is duplicate

    df.to_csv('netflix_titles_cleaned.csv', index=False)
    print(f"Saved {len(df)} rows to netflix_titles_cleaned.csv")