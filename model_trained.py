#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:18:35 2023

@author: walidkebsi
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
#------Création d'une classe Log1p Transformer pour ajouter une méthode get_param
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from joblib import dump

oil_data = pd.read_csv("oil.csv", index_col='Date', parse_dates=True)

#-----------Modèle de prédiction régression linéaire-------------------------------


# Valeur Nan normal because the rolling begins after 30 days. 
# Method :use the ewm

oil_data= oil_data.rename(columns={'Close/Last' : 'Close'})
oil_data['volatility'] = oil_data['Close'].ewm(alpha=0.6).std()

# Data cleaning 
oil_data.replace('NaN', None, inplace=True)
oil_data = oil_data.dropna(axis=0, how='any')
mean_volatility = oil_data['volatility'].mean()
oil_data['volatility'].fillna(mean_volatility, inplace=True)
oil_data = oil_data[oil_data['Close'] >= 0] #on supprime prix négatif
# Check data
print('Valeurs NaN : \n', oil_data.isnull().sum()) #check ok no NaN datas 
#Data training 
y = oil_data['Close']
X = oil_data.drop(['Close', 'Volume', 'High', 'Low'], axis=1)
#prédiction en fontion de la volatilité observé et du price d'ouverture de la session
# Encode categorical variables - no needed - only numeric data
# Normalization 
#Normalization 
# if outlier : must use RobustScaler
# Define pipelines for data preprocessing
preprocessor_linear = Pipeline([('scaler', MinMaxScaler())])
#preprocessor_tree = Pipeline([
    #('scaler', StandardScaler()),
    #('onehot', pd.get_dummies(X, columns=['Sector']))
#]). We don't need any non-linear model here.
# Define models to train
linear_model = Pipeline([
    ('preprocessor', preprocessor_linear),
    ('model', LinearRegression())
])

lasso_model = Pipeline([
    ('preprocessor', preprocessor_linear),
    ('model', Lasso())
])

ridge_model = Pipeline([
    ('preprocessor', preprocessor_linear),
    ('model', Ridge())
])

svr_model = Pipeline([
    ('preprocessor', preprocessor_linear),
    ('model', SVR())
])
models = [('Linear Regression', linear_model),
          ('Lasso Regression', lasso_model),
          ('Ridge Regression', ridge_model),
          ('SVR', svr_model)]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)

for name, model in models:
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f'{name}: mean score = {np.mean(scores):.3f}, std deviation = {np.std(scores):.3f}')

#training 

linear_model.fit(X_train, y_train)
svr_model.fit(X_train, y_train)


print('Le score du model est de :', model.score(X_test, y_test))

#X_original = scaler.inverse_transform(X_MinMax)

predictions = model.predict(X) 

print(oil_data.head())

print('Les prédictions sont les suivantes', predictions)


dump(model,'regression_model_saved.joblib')


#-----------Modèle de prédiction régression linéaire-------------------------------



