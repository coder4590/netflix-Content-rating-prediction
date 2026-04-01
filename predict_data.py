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


def predict_data():
    df=pd.read_csv('netflix_titles_cleaned.csv')

    # this is because the like of the xg boost need the categories embbeded into the number that why is this 
    le=LabelEncoder()
    df['rating_encoded']=le.fit_transform(df['rating'])

    # now we will do some of the featuer engineering and that is for now
    df['date_added'] = df['date_added'].str.strip()
    df['year_added'] = pd.to_datetime(df['date_added']).dt.year
    df['month_added'] = pd.to_datetime(df['date_added']).dt.month
    df['country_count'] = df['country'].str.split(',').str.len()
    df['actor_count'] = df['cast'].str.split(',').str.len()
    df['is_drama'] = df['listed_in'].str.contains('Drama').astype(int)
    df['genre_count'] = df['listed_in'].str.split(',').str.len()
    df['is_Action'] = df['listed_in'].str.contains('Action').astype(int)
    df['is_comedies'] = df['listed_in'].str.contains('Comedies').astype(int)
    df['duration_minutes'] = df['duration'].str.extract('(\d+)').astype(float)
    df['is_movie'] = df['duration'].str.contains('min').astype(int)
    df['seasons'] = df['duration'].str.extract('(\d+)').astype(float)
    df['is_tv_show'] = df['duration'].str.contains('Season').astype(int)
    # Length features
    df['title_length'] = df['title'].str.len()
    df['title_word_count'] = df['title'].str.split().str.len()

    # Structure features
    df['title_has_colon'] = df['title'].str.contains(':').astype(int)
    df['title_has_numbers'] = df['title'].str.contains('\d').astype(int)
    df['title_starts_with_the'] = df['title'].str.startswith('The').astype(int)

    # Content features
    df['title_contains_year'] = df['title'].str.contains('19|20').astype(int)
    df['title_has_question'] = df['title'].str.contains('\?').astype(int)

    # now we will first the make the list of the numerical and the like of the stirng column using the simple python list 
    categorical_featuure=['type']
    
    numerical_feature=['release_year','year_added','month_added','country_count','actor_count',
                       'genre_count','is_drama','is_Action','is_comedies','duration_minutes','is_movie','seasons','is_tv_show','title_length',
                       'title_word_count','title_has_colon','title_has_number','title_starts_with_the','title_contains_year','title_has_question']


    # now the next thing is that we will decide the like of the which column has the like of the is to be predicted and that is also teh main part of the data prediction and the like of the data machine leaarning also
    x=df.drop(['description','rating','show_id','date_added','rating_encoded','title', 'director', 'cast',
                'country','listed_in','duration'], axis=1)
    y=df['rating_encoded']

    x_train,x_test,y_train,y_test=train_test_split(
        x,y,
        test_size=0.2,
        random_state=42
    )

    # now the next step is the like of the conversion of the string into the like of the intger is that okay for now

    x_train=pd.get_dummies(x_train, columns=categorical_featuure, drop_first=True)
    x_test=pd.get_dummies(x_test, columns=categorical_featuure, drop_first=True)

    x_test=x_test.reindex(columns=x_train.columns, fill_value=0)

    # now the next step is thel ike of the since we have such a low number of the like of the integer row so the like of the scaling feature is not needed here 
    scaler=StandardScaler()
    available_col=[col for col in numerical_feature if col in x_train.columns]

    x_train[available_col]=scaler.fit_transform(x_train[available_col])
    x_test[available_col]=scaler.transform(x_test[available_col])

    # this is the hyper tuning of the modle using the grid search and using 9 parameter
    param_dist = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_lambda': uniform(0, 10),
        'reg_alpha': uniform(0, 5),
        'gamma': uniform(0, 5),
        'min_child_weight': randint(1, 10)
    }
    
    # Randomized search
    search = RandomizedSearchCV(
        estimator=xgb.XGBClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=10,  # Increase if you have time
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    search.fit(x_train,y_train)

    print("Best Parameters:", search.best_params_)
    print("Best CV R2:", search.best_score_)


    y_pred=search.best_estimator_.predict(x_test)

    
    accuracy=accuracy_score(y_test, y_pred)

    print(f"accuracy for xg: {accuracy:.3f}")
    print(f"Correct_prediction: {accuracy*100:.1f}")

    feature_importance = search.best_estimator_.feature_importances_
    top_features = pd.DataFrame({
    'feature': x_train.columns,
    'importance': feature_importance
    }).sort_values('importance', ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance')
    plt.title('Top 15 Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
