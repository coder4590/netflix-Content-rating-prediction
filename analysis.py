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



def analysis_data():
    df=pd.read_csv('netflix_titles.csv')

    
    # print("these are the like of the some of the little analysis on the data set which are given to us")
    print(f"Total columns are : {df.columns}")
    print(f"data types of the column are : {df.dtypes}")
    print(f"this is the like of the little description of the data : {df.describe()}")
    print(f"The total length of the data which are given to us is : {len(df)}")

    # this some of the data viuslization on the datat which is on what we are working on the data and the like of it 

    # the first thing is the like of the to see which and split the date into the like of the day and date and year and that is done using the like of the pandas and the like of this and the like of this and the like of and that it 

    year_count=df['release_year'].value_counts().head(10)

    # this is to find the like of the how much song relaase per year will pretty helpful in the like of the data analysis 
    plt.figure(figsize=(10,6))
    year_count.plot(kind='bar')
    plt.title('how much movies is released per year ')
    plt.xlabel('year')
    plt.ylabel('Count of songs')
    plt.tight_layout()
    plt.show()
    

    df['date_added']=df['date_added'].str.strip()
    df['date_added']=pd.to_datetime(df['date_added'])
    df['year']=df['date_added'].dt.year
    df['month']=df['date_added'].dt.month
    df['day']=df['date_added'].dt.day_name()

    # now we will handle the like of the and measure the all of the song tll release date to the like of the like of the to the like of the data added 

    # now the next plot is the like of the all about the like of the counting and the like of it

    top_country=df['country'].value_counts().head(10)

    # this is used ot the like of the to see how much each country is released and the like of it 
    plt.figure(figsize=(10,6))
    top_country.plot(kind='bar')
    plt.title(' top countries with the released of the movies ')
    plt.xlabel('country')
    plt.ylabel('Count of songs')
    plt.tight_layout()
    plt.show()

    # this is show which rating has most of the song released like of the now 
    top_rating=df['rating'].value_counts()
    plt.figure(figsize=(10,6))
    top_rating.plot(kind='bar')
    plt.title(' top rating with the most of the released till now  ')
    plt.xlabel('rating')
    plt.ylabel('Count of songs')
    plt.tight_layout()
    plt.show()
    
    top_director=df['director'].value_counts().head(10)
    
    # this is show the like of the which director has the highest release of the movies till now 
    plt.figure(figsize=(10,6))
    top_director.plot(kind='bar')
    plt.title(' Top Directors with the most movies released till now ')
    plt.xlabel('Director')
    plt.ylabel('Count of songs')
    plt.tight_layout()
    plt.show()

    top_countries = df['country'].value_counts().head(5)
    # this is the pie chart which is used to demonstrate and the like of the which is the ocuntry containing me the msot of the count like of the population and the like of it 
    plt.figure(figsize=(8, 8))
    plt.pie(top_countries.values, 
        labels=top_countries.index, 
        autopct='%1.1f%%',
        startangle=90)
    plt.title('Top 5 Countries by Content')
    plt.tight_layout()
    plt.show()
    df['cast'] = df['cast'].str.replace('$', 'S')

    # this is used to show the like of the which is in majority like of the tv shows or the movies that it and nothing more
    plt.figure(figsize=(8, 6))
    type_counts = df['type'].value_counts()
    type_counts.plot(kind='bar', color=['#E50914', '#221f1f'])
    plt.title('Movies vs TV Shows on Netflix')
    plt.xlabel('Content Type')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # to see which genre is like of the top rated in term of the count top 5 
    plt.figure(figsize=(12, 10))
    type_counts = df['listed_in'].value_counts().head(5)
    type_counts.plot(kind='bar')
    plt.title('Most repeated genre')
    plt.xlabel('Genre Type')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()