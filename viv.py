#Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import termcolor
import sys


#for colored text
from termcolor import colored
text=colored('MEDIUM','blue')


#splitting the dataset into dependent and independent variables
dataset = pd.read_csv('achal5.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values



#encoding column 1 as reading strings is not possible
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()




#avoiding the trap
X = X[:, 1:]




#splitting the dataset into test and training set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#training the model(object)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)



#using the object to predict y_test
y_pred = regressor.predict(X_test)





#Visualization
y1, y2, y3, y4,x1,x2 = train_test_split(y_pred, y_test,X_test[:,4], test_size = 0.005, random_state = 0)
plt.scatter(x2, y2, color  = 'blue')
plt.scatter(x2, y4, color = 'red')
plt.title('predicted vs actual values (wrt to length in kms)')
plt.xlabel('length in kms')
plt.ylabel('prediction/testing')
plt.show()

y1, y2, y3, y4,x1,x2 = train_test_split(y_pred, y_test,X_test[:,5], test_size = 0.005, random_state = 0)
plt.scatter(x2, y2, color = 'blue')
plt.scatter(x2, y4, color = 'red')
plt.title('predicted vs actual values (motorcycle congestion)')
plt.xlabel('motorcycles congestion')
plt.ylabel('prediction/testing')
plt.show()

y1, y2, y3, y4,x1,x2 = train_test_split(y_pred, y_test,X_test[:,6], test_size = 0.005, random_state = 0)
plt.scatter(x2, y2, color = 'blue')
plt.scatter(x2, y4, color = 'red')
plt.title('predicted vs actual values (cars/taxis)')
plt.xlabel('cars/taxis')
plt.ylabel('prediction/testing')
plt.show()

y1, y2, y3, y4,x1,x2 = train_test_split(y_pred, y_test,X_test[:,7], test_size = 0.005, random_state = 0)
plt.scatter(x2, y2, color = 'blue')
plt.scatter(x2, y4, color = 'red')
plt.title('predicted vs actual values (buses)')
plt.xlabel('buses')
plt.ylabel('prediction/testing')
plt.show()

y1, y2, y3, y4,x1,x2 = train_test_split(y_pred, y_test,X_test[:,8], test_size = 0.005, random_state = 0)
plt.scatter(x2, y2, color = 'blue')
plt.scatter(x2, y4, color = 'red')
plt.title('predicted vs actual values (mini vans)')
plt.xlabel('mini vans')
plt.ylabel('prediction/testing')
plt.show()

#y_pred , y_test vs x
ax=np.array([i for i in range(0,32)])
plt.plot(ax, y4, color = 'red')#ytest
plt.plot(ax,y2,color='blue')#ypred
plt.title('predicted vs actual values')
plt.xlabel('index')
plt.ylabel('performance of our model')
plt.show()

#predicting congestion based on input/real time   
new_pred=regressor.predict(np.array([[0,0,0,0,0.8,112,5000,70,200]]))
print(new_pred)
if new_pred <5000 :
    print('\033[0;32m'+'The Traffic on this Road is : LOW'+'\033[00m')
elif new_pred>= 5000 and new_pred<=20000:
    print('The Traffic on this Road is : '+text)
else:
    print('\x1b[1;31m'+'The Traffic on this Road is : HIGH'+'\x1b[0m')    







