import torch
import torch.nn as nn
import torch.nn.functional as F

import xgboost as xgb

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import PolynomialFeatures

import pandas as pds

import numpy as np

import csv

import pickle

import mlp_model_def

def module_load(model_name):
    print("Using model:", model_name)
    with open(model_name, 'rb') as f:
        global model
        model = pickle.load(f)
    return True

def model_predict(input_feature):

    poly = PolynomialFeatures(mlp_model_def.polynomial_n)

    input_feature = np.array(input_feature, dtype=np.float32).reshape(1,-1)

    x_data = poly.fit_transform(input_feature)

    x_data = model.scaler.transform(x_data)

    x_data = torch.tensor(x_data)

    y_pred = model(x_data)
    scheme_id = torch.argmax(y_pred)

    return model.search_space[scheme_id]
