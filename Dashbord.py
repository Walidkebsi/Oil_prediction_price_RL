#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 14:55:04 2023

@author: walidkebsi
"""
#Import section 
from dash import Dash 
from dash import dcc
from dash import html
import pandas as pd
import os
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from joblib import load
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from joblib import load

#---------------------------CREATION DU FICHIER VIDE PY------------------------------------
  
# Specify the path
path = '/Users/walidkebsi/Downloads/pyready_trader_go 3'
  
# Specify the file name
file = 'app2.py'
  
# Before creating
dir_list = os.listdir(path) 
print("List of directories and files before creation:")
print(dir_list)
print()
  
# Creating a file at specified location
with open(os.path.join(path, file), 'w') as fp:
    fp.write("app2.py")
    # To write data to new file uncomment
    # this fp.write("New file created")
  
# After creating 
dir_list = os.listdir(path)
print("List of directories and files after creation:")
print(dir_list)

#---------------------------CREATION DU FICHIER VIDE PY------------------------------------

#---------------------------MAIN ALGO - DASHBORD --------------------------------------------

oil_data = pd.read_csv("oil.csv", index_col='Date', parse_dates=True)
oil_data= oil_data.rename(columns={'Close/Last' : 'Close'})
oil_data['volatility'] = oil_data['Close'].ewm(alpha=0.6).std()

# Data cleaning 
oil_data.replace('NaN', None, inplace=True)
oil_data = oil_data.dropna(axis=0, how='any')
mean_volatility = oil_data['volatility'].mean()
oil_data['volatility'].fillna(mean_volatility, inplace=True)

oil_data = oil_data[oil_data['Close'] >= 0] #on supprime prix négatif

#---------------------------MAIN ALGO - DASHBORD --------------------------------------------

reg_loaded = load('regression_model_saved.joblib')

app = Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1(children="Oil Prediction Price",),
        html.P(
            children="Interactive Dashboard for oil price prediction",
        ),
        
        html.Div([
            dcc.Graph( id = 'graph_price',
                      
                      figure={ "data":[go.Scatter(x=oil_data.index, y= oil_data['Close'], mode='lines')],
                              'layout' : {'title' : "Oil Price Evolution"}})
        ]),

        html.Div([
            dcc.Graph(id='graph_volatility',
                      figure ={'data': [go.Scatter(x=oil_data.index, y = oil_data['volatility'], mode="lines")],
                              "layout" : {"title":"Volatility Evolution "}})
        ]),

        html.Div([
            html.H2('Price Prediction'),
            
            html.Div([
                html.Label('Open : '),
                dcc.Input(id='input_volatility', type='number', value=0)
            ]),
            
            html.Div([
                html.Label('Volatility : '),
                dcc.Input(id='input_Open', type='number', value=0)
            ]),
           
            
            html.Button('prediction', id='button_prediction'),
            
            html.Div(id='output_prediction')
        ])
    ]
)

def prediction_price(n_clicks, volatility, Open):
    
    X_values = [[volatility, Open]]
    Y_new = reg_loaded.predict(X_values)
    return html.Div("Forecasted price is {} USD".format(round(Y_new[0], 2)))

@app.callback(
    Output('output_prediction', 'children'),
    [Input('button_prediction', 'n_clicks')],
    [State('input_volatility', 'value'),
     State('input_Open', 'value')]
     

    
)
   

def update_output(n_clicks, volatility, Open):
    if n_clicks is None:
        return
    else:
        prediction_result = prediction_price(n_clicks, volatility,Open)
        return prediction_result
    
    
    #---------------------------MAIN ALGO - DASHBORD --------------------------------------------
#Launch server
if __name__ == "__main__":
    app.run_server(debug=True)
