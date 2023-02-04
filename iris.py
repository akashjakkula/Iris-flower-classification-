import pandas as pd
import numpy as np

data=pd.read_csv(r"C:\Users\91889\Downloads\iris.csv")

data.shape
data.size
data.info()
data.describe()

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

##from sklearn.preprocessing import MinMaxScaler
##s=MinMaxScaler()
##x=s.fit_transform(x)

##from sklearn.preprocessing import StandardScaler
##s=StandardScaler()
##x=s.fit_transform(x)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=5)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=1)
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest,ypred))

print(model.predict([[7.1,3.1,2.5,0.4]]))














#######fill the empty spaces or nan or null values
######import pandas as pd
######data=pd.read_csv(r"C:\Users\91889\Downloads\iris.csv")
######data.shape
######data.size
######data.info()
######data.describe()
#######finding the mean of the pw colomn
######mean=data['pw'].mean()
#######filling the nan to mean value
######data.fillna(mean,inplace=True)
#######rechecking...
######data.info()
######data.describe()
