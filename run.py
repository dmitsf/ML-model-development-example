#!/usr/bin/env python
# coding: utf-8

# Model Developed
# All building steps are described in a Report
import os
import sys
from time import time
from math import sqrt
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

SAVE_PATH = '/home/df/models'

# I move the saved model to the same directory where all the files are located 
LOAD_PATH = '.' # '/home/df/models'
MODEL_NAME = 'model.pkl'


def read_data(filename):
    return pd.read_csv(filename)


def build_model(X, y, params={'bootstrap': False, 'max_depth': None,
                              'max_features': 'sqrt',
                              'min_samples_leaf': 1, 'min_samples_split': 4,
                              'n_estimators': 10}, # 10 estimators to decrease size of model
                model=RandomForestRegressor):

    # Model with Data Processing Pipeline
    # I chose RFR regressor with optimized hyperparameters.
    # Params were found with a help of Random and Grid search.
    imputer = SimpleImputer(strategy='mean')
    regressor = model(**params, n_jobs=-1)
    model = Pipeline([('imputing', imputer),
                      ('regression', regressor)])

    t = time()
    model.fit(X, y)
    print("Overall fitting time: %.3f sec." % (time() - t))

    return model


def evaluate_model(model, X, y, X_test, y_actual):
    print('Model Performance:')
    print('    Train score: %.3f' % model.score(X, y))
    evaluate_model_on_test_dataset(model, X_test, y_actual)


def evaluate_model_on_test_dataset(model, X_test, y_actual):
    print("    Test score: %.3f" % model.score(X_test, y_actual))
    y_predicted = model.predict(X_test)
    rms = sqrt(mean_squared_error(y_actual, y_predicted))
    print("    RMSE: %.3f" % rms)

    correct_predictions = 0
    for y_a, y_p in zip(y_actual, y_predicted):
        if abs(y_a - y_p) <= 3:
            correct_predictions += 1
    print("    Correct predictions: %.3f %%" % (correct_predictions / len(y_actual) * 100))

    errors = abs(y_predicted - y_actual)
    mape = 100 * np.mean(errors / y_actual)
    accuracy = 100 - mape
    print('    Average Error: %.3f' % np.mean(errors))
    print('    Accuracy: %.3f' % accuracy)
    

def save_model(model):
    with open(os.path.join(SAVE_PATH, MODEL_NAME), 'wb') as f:
        pickle.dump(model, f)


def load_model():
    with open(os.path.join(LOAD_PATH, MODEL_NAME), 'rb') as f:
        return pickle.load(f)


# Code adapted to load dataset and show quality metrics
if __name__ == "__main__":
    # Check number of arguments:
    # if no filename, use default
    if len(sys.argv) < 2:
        filename = "dataset_00_with_header.csv"
    else:
        filename = sys.argv[1]

    # Mode to train model: I expect here 'train'
    if len(sys.argv) > 2:
        mode = sys.argv[2]
    else:
        mode = 'evaluate'

    # Read the dataset
    try:
        data = read_data(filename)
    except Exception as e:
        raise RuntimeError("Error while reading dataset") from e

    if mode == 'train':
        # Split the dataset into train and test
        train, test = train_test_split(data, test_size=0.2)
        X = train.drop("y", 1)
        y = train.y

        # Build model
        model = build_model(X, y)
        save_model(model)

        X_test = test.drop("y", 1)
        y_actual = test.y

        # Evaluate constructed model
        evaluate_model(model, X, y, X_test, y_actual)

    else:
        model = load_model()

        X_test = data.drop("y", 1)
        y_actual = data.y

        # Evaluate constructed model
        evaluate_model_on_test_dataset(model, X_test, y_actual)
