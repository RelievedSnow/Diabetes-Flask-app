# Diabetes-Flask-app
Diabetes application using machine learning and python flask.

# Working of the App.
1. Creating the Model.
* First we create a Machine Learning Model
* We Collect The Data and Pre-process the Data
* We load the data into classification model.
* We create a prediction system where '1' means 'Diabetic' and 'O' mean 'Non-Diabetic'.
* We then dump the now trained 'model(classifier)' and 'scaler function()' that is used to bring the values to similar range and create a '.joblib' file.

2. Creating our Flask app.
* We import the required libraries.
1. from flask import Flask, render_template, request
2. import numpy as np
3. from joblib import load
* We create a '/home' route where we take input from users.
* The 'POST' request is made to the server in the form of string and the data gets stored in the variables.
* We store the strings in the form of list.
* We pass the elements of the list into a numpy array for the model to process it easily.
* We then convert the '1-Dimentional' numpy array into '2-Dimentiona'l so that the model can process it.
* We Load the model and scaler '.joblib' files. These files contain the trained model.
* We scale the input received from the users as the values of each feature is different and not similar. We need similar type of values for the model to predict the outcome correctly.
* Now we make prediction using the '.predict()' funtion.
* If the prediction is '1' the person is 'diabetic' else he's 'non-diabetic'.

3. Html pages (index.html & result.html)
* In the 'index.html' page, we create a form that takes input from users in the form of text and return it to the server.
* We create a Predict button that submits the data to the server as a 'POST' request.
* After the prediction is made, it is rendered to the 'result.html' page on the '/home' route.
* The 'result.html' page consists of a card that displays the outcome of the prediction.
* To predict again you can press the Predict Again button that return the user back to the '/home' route.

4. How to RUN the app.
* Open the terminal of your vscode and type 'pyhton run' or 'python .\app.py'

  
# Home Page.
![Screenshot (117)](https://github.com/user-attachments/assets/2cfdcbbf-335c-48c3-8206-62df1dcdd09d)
![Screenshot (118)](https://github.com/user-attachments/assets/e4db338c-fa1b-46a8-8f30-2508c5056fcd)
![Screenshot (119)](https://github.com/user-attachments/assets/2aa07c7d-9619-4e0d-9651-c1f00da673e4)
![Screenshot (120)](https://github.com/user-attachments/assets/1aec072f-0dfd-4291-823d-eb2277836a9a)

# Result Page
![Screenshot (121)](https://github.com/user-attachments/assets/12f32a24-4cf4-4d89-a9cf-6d50e3e0c395)

