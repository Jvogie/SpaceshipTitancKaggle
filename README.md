# Spaceship Titanic Kaggle Competition

This project is focused on predicting whether a passenger will be transported or not, using different machine learning algorithms. All data came from Spaceship Titanic Kaggle Competition


## Overview

This repository contains a Python script that does the following:

- Reads data from CSV files (`train.csv` and `test.csv`).
- Drops irrelevant features ('PassengerId', 'Name') from the datasets.
- Handles missing values by applying SimpleImputer for categorical columns and KNNImputer for numerical columns.
- Performs feature engineering by splitting the 'Cabin' feature into two new features 'Deck' and 'Side'.
- One hot encodes the categorical features for both training and testing datasets.
- Scales the numerical features using a StandardScaler.
- Trains four different models on the training data: Random Forest, XGBoost, XGBoost Random Forest, and a Neural Network.
- Evaluates each model using a validation set.
- Applies the trained models to the test data to generate predictions.
- Writes the predictions for each model into separate CSV files (`rf_submission.csv`, `xgb_submission.csv`, `xgbrf_submission.csv`, `nn_submission.csv`).

## Results
After submitting each of the submission files to Kaggle the results (% accurate) were:

XGB - 79.214  
XGBRF - 79.167  
NN - 78.816  
RF - 78.7

## Requirements

The script uses several Python libraries including Pandas, Numpy, Matplotlib, TensorFlow, Keras, Scikit-learn, and XGBoost. The TensorFlow library is used to construct a neural network, while the Keras library is used for hyperparameter tuning.

## How to Run

To run the script, you will need Python installed on your machine. Execute the script using any Python interpreter. You need to have the data files (`train.csv` and `test.csv`) in the same directory as the script.

The script will read in the training and test data, preprocess and transform the data, train the models, make predictions, and save these predictions into separate CSV files.

## Note

The Keras tuner section is commented out in the code. This part of the code was used to find the best hyperparameters for the Neural Network. The optimal hyperparameters found using this method were then hardcoded into the Neural Network model definition. Uncomment this section if you want to perform hyperparameter tuning.

## Left Over Thoughts

I was mainly focused on trying to make a neural network so I spent the vast majority of my time working on that and didn't mess around too much with the XGB,
XGBRF, or RF models.
