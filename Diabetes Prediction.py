# importing dependencies

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from joblib import dump, load
# Data Collection and Data Pre-Processing

# storing dataset into the diabeties_data var using .read_csv
diabetes_dataset = pd.read_csv('C:/Users/DELL/PycharmProjects/MLprojects/diabetes.csv')

# reading 1st five rows of the dataset
# print(diabetes_dataset.head())

# checking the no. of rows and columns in the dataset
# print(diabetes_dataset.shape)  # 768 different data with 9 cols

# statistical data
# print(diabetes_dataset.describe())  # 25%,50%,75% percentile mean that there are 25% values are present than the given value in each feature

# checking how many 1's and how many 0's (1-> diabetic, 0-> non-diabetic)
# print(diabetes_dataset['Outcome'].value_counts())  # 0 = 500, 1 = 268

# finding mean value for diabetic and non-diabetic patients
# print(diabetes_dataset.groupby('Outcome').mean())

# separating labelled values from numerical values

X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# print(X, Y)

# Data Standardization (to make all the values in each column in a particular range)

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

# OR

# standardized_data = scaler.fit_transform(X)
# print(standardized_data)

# Again storing the new standardized values into X
X = standardized_data

# split the data into training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# print(X.shape, X_train.shape, X_test.shape)

# Training Model (We are using Support Vector Model)
classifier = svm.SVC(kernel='linear')  # loading the support vector model into classifier var

# training the support vector machine
classifier.fit(X_train, Y_train)  # training data and training label data

# check the accuracy of the training model

X_train_prediction = classifier.predict(X_train)
train_data_prediction = accuracy_score(X_train_prediction, Y_train)
# print(train_data_prediction)

# check the accuracy of test model
X_test_prediction = classifier.predict(X_test)
test_data_prediction = accuracy_score(X_test_prediction, Y_test)
# print(test_data_prediction)

# Making Prediction System
input_data = (6,148,72,35,0,33.6,0.627,50)

# converting the data in the input var into numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshaping the input_data value from 1-D to 2-D
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# standardized the input data as we cannot use the raw values because we have feed the model standardized values
std_data = scaler.transform(input_data_reshaped)

prediction = classifier.predict(std_data)
# print(prediction)

# if prediction[0] == 0:  # if the first value in the list is
#     print("Non-Diabetic")
# else:
#     print("Diabetic")

# Save the model and scaler
dump(classifier, 'classifier.joblib')
dump(scaler, 'scaler.joblib')

