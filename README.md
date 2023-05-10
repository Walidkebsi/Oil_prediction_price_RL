# Oil_prediction_price_RL
Machine Learning prediction - Oil Close Price


Oil prediction model (cross validation and normalization included)

STEPS TO USE MY DASHBOARD

1# Download Dash in your terminal (mac : pip3 install dash) 2# Download the oil.csv file in the right repertory (coding environnement) 3# Run the Dashboard.py file in your console

Warning :If overfitting : the solution is to merge a new dataframe with additionnal datas to train the model with other datas. Nevertheless, you can see my crossvalidation. Among a set of linear model, the best model was the LinearRegression (std deviation approximately equal to 0.007 and mean score = 0.986

Linear Regression score (test_set) = 0.994

I am working on a calibration of the Heston model using NNs. Next project is to integrate my Heston model calibrated to predict implied volatility surfaces on an european option. Then, I will be able to predict the price of an option in fonction of its maturity and strike for option smile dependents.

-- Imperial College London --
