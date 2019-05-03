# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
import pandas as pd
import sys
import os

# obtain real path
pathname = os.path.dirname(sys.argv[0])

# Importing the dataset
dataset = pd.read_csv(pathname + '/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()


# Avoiding the Dummy Variable Trap
# take all columns  :, start from 1->1:
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Using backward elimination for an optimal model
# put the array that we need first so then add the matrix, in this case it will stay a the first column
X = np.append(arr=np.ones((50, 1)).astype(int), values=X , axis=1)

# X_opt = X[:, [0,1,2,3,4,5]]
# # create a new regresor
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# print('*********************************************************************')
# print(regressor_OLS.summary())
# print('*********************************************************************')
#
# # now check the result of  P>|t| columns and delete the variable with highest value
# X_opt = X[:, [0,1,3,4,5]]
# # create a new regresor
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# print('*********************************************************************')
# print(regressor_OLS.summary())
# print('*********************************************************************')
#
# # now check the result of  P>|t| columns and delete the variable with highest value
# X_opt = X[:, [0,3,4,5]]
# # create a new regresor
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# print('*********************************************************************')
# print(regressor_OLS.summary())
# print('*********************************************************************')
# # now check the result of  P>|t| columns and delete the variable with highest value
#
# X_opt = X[:, [0,3,5]]
# # create a new regresor
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# print('*********************************************************************')
# print(regressor_OLS.summary())
# print('*********************************************************************')
# # now check the result of  P>|t| columns and delete the variable with highest value
#
#
# X_opt = X[:, [0,3]]
# # create a new regresor
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# print('*********************************************************************')
# print(regressor_OLS.summary())
# print('*********************************************************************')

# this is the last iteration because const and x1 are below the 0.05% that were 
# consider as good performance
# optimal team . stronget impact with x3 independent variable->R&D Spend
# so that would be the selected variable


# automating
import statsmodels.formula.api as sm


def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())
    return x


SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
print('variables that best fit..->', X_Modeled)


# Backward Elimination with p - values and Adjusted R Squared:
import statsmodels.formula.api as sm

def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50, 6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:, j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:, [0, j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print(regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x


# SL = 0.05
# X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# X_Modeled = backwardElimination(X_opt, SL)

