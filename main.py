import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier,XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow import keras
from tensorflow.python.keras import layers

from keras_tuner.tuners import RandomSearch

from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# dropping passengerid and name as they have no relationship to the object being transported
train_data.drop(['PassengerId', 'Name'], axis=1, inplace=True)
test_data.drop(['PassengerId', 'Name'], axis=1, inplace=True)

# -------------------------------
# Feature Engineering
# -------------------------------

# imputer methods
sm = SimpleImputer(strategy='most_frequent')  # method used for Cateogrical values
knn = KNNImputer()

# categorizing columns by class of data
categorical_columns = test_data.select_dtypes(include='object').columns
numerical_columns = test_data.select_dtypes(exclude='object').columns

# applying imputers to both sets of data

train_data[categorical_columns] = sm.fit_transform(train_data[categorical_columns])

test_data[categorical_columns] = sm.fit_transform(test_data[categorical_columns])

train_data[numerical_columns] = knn.fit_transform(train_data[numerical_columns])

test_data[numerical_columns] = knn.fit_transform(test_data[numerical_columns])

# Splitting Cabin column into deck and side columns for both data sets

train_data["Deck"] = train_data['Cabin'].str.split('/').str[0]
train_data["Side"] = train_data['Cabin'].str.split('/').str[2]

test_data["Deck"] = test_data['Cabin'].str.split('/').str[0]
test_data["Side"] = test_data['Cabin'].str.split('/').str[2]

train_data.drop(['Cabin'], axis=1, inplace=True)
test_data.drop(['Cabin'], axis=1, inplace=True)

# Convert Transported column values from T/F to 1/0
train_data['Transported'] = train_data['Transported'].map({True: 1, False: 0})

# one hot encode both data sets. Drop first column from each time one hot encoding is done
# in order to save space and time
train_encoded = pd.get_dummies(train_data, drop_first=True, dtype=int)
test_encoded = pd.get_dummies(test_data, drop_first=True, dtype=int)

# print(train_encoded.head())
# print(test_encoded.head())

# ---------------------------------
# Model Testing
# ---------------------------------

X= train_encoded.drop('Transported',axis = 1)
y = train_encoded['Transported']

x_train,x_val,y_train,y_val = train_test_split(X,y, test_size = 0.23 ,random_state = 42)

# create a scaler object
scaler = StandardScaler()

# fit and transform the training data
x_train[numerical_columns] = scaler.fit_transform(x_train[numerical_columns])

# remember to only transform the validation data
x_val[numerical_columns] = scaler.transform(x_val[numerical_columns])

# scale test data
test_encoded[numerical_columns] = scaler.transform(test_encoded[numerical_columns])

# Random Forest Model
def RandomForest(x_train, y_train, x_val, y_val):
    # create a random forest classifier
    clf = RandomForestClassifier(n_estimators=100)

    # train the classifier
    clf.fit(x_train, y_train)

    # predict on validation set
    y_pred = clf.predict(x_val)

    # calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)

    return accuracy, clf

# XGBoost Model
def XGBoost(x_train, y_train, x_val, y_val):
    # create a XGBoost classifier
    clf = XGBClassifier(n_estimators=100, eval_metric='logloss')

    # train the classifier
    clf.fit(x_train, y_train)

    # predict on validation set
    y_pred = clf.predict(x_val)

    # calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)

    return accuracy, clf

# XGBoost Random Forest Model
def XGBRF(x_train, y_train, x_val, y_val):
    # create a XGBRF classifier
    clf = XGBRFClassifier(n_estimators=100, eval_metric='logloss')

    # train the classifier
    clf.fit(x_train, y_train)

    # predict on validation set
    y_pred = clf.predict(x_val)

    # calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)

    return accuracy, clf

# Neural Network

# Using Keras tuner to find best hyperparameters for NN
def build_model(hp):
    model = tf.keras.Sequential()

    # Add an input layer
    #model.add(layers.Input(shape=(x_train.shape[1],)))
    model.add(layers.Dense(units=hp.Int('units_0', 32, 512, 32),
                           activation='relu',
                           input_shape=(x_train.shape[1],)))
    # Tune the number of hidden layers and neurons per layer
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i), 32, 512, 32), activation='relu'))

    # Add the output layer
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# tuner = RandomSearch(
#     build_model,
#     objective='val_accuracy',
#     max_trials=10,
#     directory='my_dir',
#     project_name='my_project'
# )
#
# tuner.search(x_train, y_train, epochs=100, validation_data = (x_val,y_val))

# Get the best hyperparameters
# best_hp = tuner.get_best_hyperparameters(1)[0]




def NeuralNet(x_train, y_train, x_val, y_val):
    # Define model architecture
    model = tf.keras.Sequential()

    # Add first hidden layer with input shape
    model.add(layers.Dense(units=480,
                           activation='relu',
                           input_shape=(x_train.shape[1],)))

    # Adding additional layers based on the tuned parameters
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dense(units=256, activation='relu'))
    model.add(layers.Dense(units=192, activation='relu'))
    model.add(layers.Dense(units=352, activation='relu'))

    # Add the output layer
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  metrics=['accuracy'])

    # Fit the model
    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1)

    # evaluate the model
    _, accuracy = model.evaluate(x_val, y_val, verbose=0)

    return accuracy, model

# Run each model and print out accuracy

# print("Random Forest Accuracy: ", RandomForest(x_train, y_train, x_val, y_val))
# print("XGBoost Accuracy: ", XGBoost(x_train, y_train, x_val, y_val))
# print("XGBRF Accuracy: ", XGBRF(x_train, y_train, x_val, y_val))
# print("Neural Network Accuracy: ", NeuralNet(x_train, y_train, x_val, y_val))

# functions to apply models to a given data set

# Random Forest Model
def apply_RandomForest(model, x_test):
    # predict on test set
    predictions = model.predict(x_test)
    return predictions

# XGBoost Model
def apply_XGBoost(model, x_test):
    # predict on test set
    predictions = model.predict(x_test)
    return predictions

# XGBoost Random Forest Model
def apply_XGBRF(model, x_test):
    # predict on test set
    predictions = model.predict(x_test)
    return predictions

# Neural Network
def apply_NeuralNet(model, x_test):
    # predict on test set
    predictions = model.predict(x_test)
    return predictions

# Apply functions to test data and get prediciton results

rf_predictions = apply_RandomForest(RandomForest(x_train, y_train, x_val, y_val)[1], test_encoded)
xgb_predictions = apply_XGBoost(XGBoost(x_train, y_train, x_val, y_val)[1], test_encoded)
xgbrf_predictions = apply_XGBRF(XGBRF(x_train, y_train, x_val, y_val)[1], test_encoded)
nn_predictions = apply_NeuralNet(NeuralNet(x_train, y_train, x_val, y_val)[1], test_encoded)


#create CSV files for each of the model predictions with passengerId attached

sub_data=pd.read_csv('test.csv')

# Random Forest
rf_submission = pd.DataFrame({
    "PassengerId": sub_data["PassengerId"],
    "Transported": rf_predictions
})
rf_submission['PassengerId'] = rf_submission['PassengerId'].astype(str)
rf_submission['Transported'] = rf_submission['Transported'].map({0:False,1:True})
rf_submission.to_csv('rf_submission.csv', index=False)

# XGBoost
xgb_submission = pd.DataFrame({
    "PassengerId": sub_data["PassengerId"],
    "Transported": xgb_predictions
})
xgb_submission['PassengerId'] = xgb_submission['PassengerId'].astype(str)
xgb_submission['Transported'] = xgb_submission['Transported'].map({0:False,1:True})
xgb_submission.to_csv('xgb_submission.csv', index=False)

# XGBoost Random Forest
xgbrf_submission = pd.DataFrame({
    "PassengerId": sub_data["PassengerId"],
    "Transported": xgbrf_predictions
})
xgbrf_submission['PassengerId'] = xgbrf_submission['PassengerId'].astype(str)
xgbrf_submission['Transported'] = xgbrf_submission['Transported'].map({0:False,1:True})
xgbrf_submission.to_csv('xgbrf_submission.csv', index=False)

# Neural Network
nn_submission = pd.DataFrame({
    "PassengerId": sub_data["PassengerId"],
    "Transported": nn_predictions.flatten()
})
nn_submission['PassengerId'] = nn_submission['PassengerId'].astype(str)
nn_submission['Transported'] = nn_submission['Transported'].apply(lambda x: True if x > 0.5 else False)
nn_submission.to_csv('nn_submission.csv', index=False)