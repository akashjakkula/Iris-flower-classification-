##Iris flower classification is a very popular machine learning project.
##The iris dataset contains three classes of flowers, Versicolor, Setosa, Virginica.
##Each class contains 4 features, ‘Sepal length’, ‘Sepal width’, ‘Petal length’, ‘Petal width’.
##The aim of the iris flower classification is to predict flowers based on their specific features.

import pandas as pd
import numpy as np

data=pd.read_csv(r"C:\Users\91889\Downloads\iris.csv")

#Gives shape of the data.
data.shape

#Gives size of the data.
data.size

#Gives information about the data.
data.info()
data.describe()

#Allocating dependent and target variables as x and y.
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

#By using MinMaxScaler, we can re-arrange the values in range of [0,1].
from sklearn.preprocessing import MinMaxScaler
s=MinMaxScaler()
x=s.fit_transform(x)

#By using train_test_split, the algorithm will divides the data as training data and testing data.
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=5)

#here we are using KNN algorithm.
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=1)

#The model has been desgined by the KNN algorithm.
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)

#By using accuracy_score, we can know the accuracy of the prediction.
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)

#By using confusion_matrix, we can know the matrix between the test values and predictied values.
from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest,ypred))

#The algorithm has been trained and its predicting the BUSSINESS PROBLEM.
print(model.predict([[7.1,3.1,2.5,0.4]]))
