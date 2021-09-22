#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog

file = filedialog.askopenfilename()
dst = pd.read_csv(file)
print (dst.head())

print (dst.shape)
print (dst.drop_duplicates(inplace = True))
print (dst.shape)

print (dst.isnull().sum())
dataset = dst.dropna(axis=1)
print (dataset['Outcome'].value_counts())

import seaborn as sns
sns.countplot(dataset['Outcome'], label = 'Count')
plt.show()
print (dataset.dtypes)

sns.pairplot(dataset.iloc[:,0:5])
plt.show()

#Showing correlation between the columns within the dataset
sns.heatmap(dataset.iloc[:,0:8].corr())
plt.show()
             
data = dataset.values
print(data)



X = data[:,0:8]
Y = data[:,8]

print (X)
print (Y)

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
#The min_max scaler scales the data so that all the input features or independent variables lies betwen 0 and 1  
#This is to allow the model make its predictions 
X_scale = min_max_scaler.fit_transform(X)
print(X_scale )


#This code split the dataset into 80% training and 25% testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size = 0.2, random_state = 4)

print()
print("======================================================================== ")
print("--------------BUILDING THE ANN PREDICTION MODEL--------------------------")
print("======================================================================== ")

#Building the ANN Model
import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

modelANN = Sequential([
    Dense(12, activation = 'relu', input_shape=(8,)),
    Dense(15, activation = 'relu',),
    Dense(1, activation = 'sigmoid')
    ])
#This code will be compilling the module
modelANN.compile(
    optimizer ="sgd", #Stochastic gradient descent
    loss = 'binary_crossentropy', #This is use for binary classification
    metrics = ['accuracy']
    )

#This code visualizes the training accuracy and the validation accuracy to see if the model is overfitting    
hist = modelANN.fit(X_train, Y_train, batch_size = 70, epochs = 500, validation_split = 0.2)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper right')
plt.show()

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper left')
plt.show()


#Make prediction and print the actual values
prediction = modelANN.predict(X_test)
prediction = [1 if Y>=0.5 else 0 for Y in prediction]
print(prediction)
print(Y_test)

#Evaluate the model on the training dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  
pred = modelANN.predict(X_train)
pred = [1 if Y>= 0.5 else 0 for Y in pred]
print(classification_report(Y_train, pred))
print("Confusion Matrix: \n", confusion_matrix(Y_train, pred))
print()
print('ANN Training Accuracy: ', accuracy_score(Y_train, pred))

#Evaluate the model on the test dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  
pred = modelANN.predict(X_train)
pred = [1 if Y>= 0.5 else 0 for Y in pred]
print(classification_report(Y_train, pred))
print("Confusion Matrix: \n", confusion_matrix(Y_train, pred))
print()
print('ANN Testing Accuracy: ', accuracy_score(Y_train, pred))

print()
print("======================================================================== ")
print("------------------------BUILDING THE SVM MODEL---------------------------")
print("======================================================================== ")
from sklearn import svm
modelSVM = svm.SVC(kernel='linear')
modelSVM.fit(X_train, Y_train) #training the model on X_train and Y_train of the dataset
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
print(classification_report(Y_test, modelSVM.predict(X_test)))
print('SVM traing accuracy is: ', modelSVM.score(X_train, Y_train))
print('Support_Vector_Machine Testing Accuracy: ', accuracy_score(Y_test, modelSVM.predict(X_test)))
#Here the code will be printing the model accuracy on the training data
print()
print('************************************************************')
