#--------------------------------------------ABOUT DATASET
#Context
#Machine Learning with R by Brett Lantz is a book that provides an introduction to machine learning using R. 
# As far as I can tell, Packt Publishing does not make its datasets available online unless you buy the book and create a user account which can be a problem if you are checking the book out from the library or borrowing the book from a friend.
# All of these datasets are in the public domain but simply needed some cleaning up and recoding to match the format in the book.


#----------------------------------------WORKFLOW
#data collection
#data analysis
#data pre-processing
#train-test-data
#model use
#predict our data 

#--------------------------------------IMPORT LABRary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics#this is used becuse out put values are not two or three more than that

#----------------------------------------data collection
data = pd.read_csv("C:/Users/kunde/all vs code/ml prject/insurance.csv")
print(data.head(5))
print(data.columns)
print(data.tail(5))
print(data.shape)
print(data.info())
print(data.describe())
print(data["sex"].value_counts())
print(data["children"].value_counts())
print(data["smoker"].value_counts())
print(data["region"].value_counts())
data.replace({"sex":{"female":0, "male":1}, "smoker":{"no":0, "yes":1}, "region": {"southeast":0, "southwest":1, "northwest":2, "northeast":3}}, inplace=True)
print(data.head(5))
#---------------------------------------separtion of data
x = data.drop(columns=["charges"], axis=1)
y = data["charges"]

print(x.head(5))
#print(y)

#--------------------------------------train-test-split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print(x.shape, x_train.shape, x_test.shape)

#--------------------------------------model use 
model = RandomForestRegressor()
model.fit(x_train, y_train)

#-----------------------------------------prediction of data train data
y_train_p = model.predict(x_train)
print(y_train_p, "this is our dataprediction", y_train, "this is true dataset")
accur = metrics.r2_score(y_train, y_train_p)

#------------------------------------------prediction test data
y_test_p = model.predict(x_test)
print(y_test_p, "this is our prediction of data", y_test, "this is true data values")
accur = metrics.r2_score(y_test, y_test_p)
print(accur, "this is score of test model")

#----------------------------------------new data prediction 
x = [19  ,  0 , 27.900  ,       0   ,    1   ,    1]
x_new = np.asarray(x)
print(x)
x_ = x_new.reshape(1, -1)
y_pr = model.predict(x_)
print(y_pr)





