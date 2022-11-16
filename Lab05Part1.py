#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:00:31 2022

@author: keithcheng
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('train.csv')

#plt.scatter(df.GrLivArea,df.SalePrice)

Train_X = df[['GrLivArea']]
Train_y = df[['SalePrice']]


# Split the data into training/testing sets
Train_X_train = Train_X[:-292]
Train_X_test = Train_X[-292:]

# Split the targets into training/testing sets
Train_y_train = Train_y[:-292]
Train_y_test = Train_y[-292:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(Train_X_train, Train_y_train)

# Make predictions using the testing set
Train_y_pred = regr.predict(Train_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Train_y_test, Train_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(Train_y_test, Train_y_pred))



#plt.scatter(Train_X_test, Train_y_test)
#Plot outputs
plt.scatter(Train_X_test, Train_y_test, color="black")
plt.plot(Train_X_test, Train_y_pred, color="blue")

plt.xticks(())
plt.yticks(())
print(regr.score(Train_X_test,Train_y_test))

#plt.show()

df['residual']= Train_y_test - Train_y_pred
print("the residual's mean is ",np.mean(df['residual']))
numeric_features = df.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print (corr['SalePrice'].sort_values(ascending=False)[:10], '\n') #top 10


