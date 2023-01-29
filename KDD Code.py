# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 23:52:46 2022

@author: 9941064513.UPS
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import sklearn.metrics as ms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv("data.csv",on_bad_lines="warn",names=[i for i in range(41)],engine=("python"), chunksize=1000)
df = pd.concat(data)

print(df.info())
print(df.isnull().sum())

X= df.iloc[:,:-1].values
y= df.iloc[:,40].values

#X1=pd.get_dummies(X[:,1],drop_first=True)
#X1=pd.get_dummies(X[:,2],drop_first=True)
#X1=pd.get_dummies(X[:,3],drop_first=True)

#X=np.delete(X,[1,2,3], axis=1)

#lable Encoder for independent value y

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

#lable Encoder for dependent value X

labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0])
X[:,1] = labelencoder.fit_transform(X[:,1])
X[:,2] = labelencoder.fit_transform(X[:,2])

# OR you can use the below line to be easear than above lines
# "df=pd.get_dummies(df,columns=[1,2,3].values"

#Train Test Split

from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#Standarization 

from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()

X_train = standardscaler.fit_transform(X_train)
X_test = standardscaler.fit_transform(X_test)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 40, random_state = 42)
regressor.fit(X_train, y_train)

y_pred1=regressor.predict(X_test)



# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state = 42)
LR.fit(X_train, y_train)

y_pred2 = LR.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred2)
print(cm)

# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
reg= DecisionTreeRegressor(random_state = 42)
reg.fit(X_train, y_train)

y_pred3=reg.predict(X_test)


from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
regressor.fit(X_train, y_train)

y_pred4 = regressor.predict(X_test)


print('explained_variance_score for random forest',ms.explained_variance_score(y_test, y_pred1))
print('explained_variance_score for logistic regression',ms.explained_variance_score(y_test, y_pred2))
print('explained_variance_score desision tree',ms.explained_variance_score(y_test, y_pred3))
print('explained_variance_score SVR',ms.explained_variance_score(y_test, y_pred4))
