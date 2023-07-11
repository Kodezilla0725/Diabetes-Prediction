# Diabetes-Prediction

## Introduction
This project utilises ML in Python to accurately predict Diabetes using patient data, aiding in early disease detection and identifying high-risk individuals. It is invaluable in resource-limited settings, improving health outcomes, reducing costs, and enhancing quality of life for individuals and communities.

In modern healthcare, non-invasive diagnostic tests have become a crucial requirement, but the significant volume of electronic data poses challenges for accurate diagnosis of diseases. Supervised machine learning algorithms have been proven to outperform traditional diagnostic systems, aiding in the early detection of high-risk diseases. This project aims to develop a cost effective diagnostic system that utilizes fundamental health parameters to predict potential future health complications using machine learning models. The proposed system can be used as a reliable tool for early detection and prevention of chronic diseases, particularly in resource-limited settings. The integration of machine learning algorithms in healthcare has the potential to revolutionize the medical industry, enhance patient outcomes, and reduce healthcare costs.

## 4.2 Diabetes Model Implementation

### 4.2.1 Importing required Dataset and Libraries

The code imports the necessary libraries such as NumPy, Pandas, SVM, and accuracy metrics from Scikit-Learn. NumPy is used for mathematical calculations, Pandas for data manipulation, and Scikit-Learn for machine learning algorithms.

The code loads the diabetes dataset using the `read_csv()` function from Pandas. This function reads the data from a CSV file and stores it in a Pandas DataFrame.

### 4.2.2 Splitting Data into Training & Test Data and Model Training

The code splits the data into training and testing sets using the `train_test_split()` function from Scikit-Learn. The function takes in the input and output variables `X` and `Y`, and splits the data into training and testing sets based on the `test_size` parameter. The `stratify` parameter is set to `Y` to make sure that the proportion of labels in the training and test sets is the same as the proportion of labels in the whole dataset. The `random_state` parameter is set to a fixed value to ensure reproducibility of the results.

The code creates an instance of the SVM classifier using `svm.SVC()`. The `kernel` parameter is set to 'linear' which specifies the linear kernel for the SVM model. The SVM classifier is a binary classifier that separates the data into two classes based on a linear decision boundary.

The code trains the SVM model on the training data using the `fit()` method of the SVM classifier. The `fit()` method takes in the training input and output variables `X_train` and `Y_train` respectively.

### 4.2.3 Testing

The code evaluates the accuracy of the model on the training data by predicting the output using the `predict()` method of the SVM classifier on the training input `X_train` and `X_test`. The predicted output is compared to the actual output `Y_train` and `Y_test` respectively using the `accuracy_score()` function from Scikit-Learn. The `accuracy_score()` function returns the accuracy of the model on the training and testing data.

### 4.2.4 Result Analysis

The code creates a new input as a tuple and stores it in the `input_data` variable. The input data consists of the following features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age. The code uses the `predict()` method of the trained SVM classifier to make a prediction for the input data. The `predict()` method takes in the input data in the form of a NumPy array and returns the predicted output. The code prints the prediction as 0 or 1, where 0 indicates that the person is not diabetic and 1 indicates that the person is diabetic.
