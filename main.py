import imp
import os
import psutil
import numpy as np  # import auxiliary library, typical idiom
import pandas as pd  # import the Pandas library, typical idiom
from pandas import read_csv
import statsmodels.api as sm
import time
import pm4py
from datetime import datetime
from datetime import date
from datetime import datetime
from datetime import timedelta
from sklearn.feature_selection import SelectKBest
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from numba import jit

from sklearn.linear_model import LinearRegression  # for linear regression
from sklearn import linear_model
from sklearn.cluster import KMeans  # for clustering
from sklearn.tree import DecisionTreeClassifier  # for decision tree mining
from sklearn.metrics import mean_absolute_error, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot
from matplotlib import pyplot
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dropout
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

file_export = 'export2018.csv'
data = pd.read_csv(file_export)

data = data.sort_values(by=['case', 'startTime'])


#Duration
@jit(parallel=True)
def calculator_nb(case, startTime):
    res = np.empty(len(case), dtype=object)
    idx = 0
    for _ in case:
        if (idx + 1 >= len(case)):
            break

        if (case[idx + 1] == case[idx]):
            res[idx] = startTime[idx + 1]
        else:
            res[idx] = startTime[idx]

        idx += 1
    return res


data['completeTime'] = calculator_nb(data['case'].values,
                                     data['startTime'].values)
data.at[317373, 'completeTime'] = data.at[317373, 'startTime']

data['startTime'] = pd.to_datetime(data['startTime'])
data['completeTime'] = pd.to_datetime(data['completeTime'])
data['duration'] = data['completeTime'] - data['startTime']
#to turn duration into seconds:
duration = data['duration']
duration = duration / np.timedelta64(1, 's')
data['duration'] = duration


#Next event
@jit(parallel=True)
def calculator_nb(case, event):
    res = np.empty(len(case), dtype=object)
    idx = 0
    for _ in case:
        if (idx + 1 >= len(case)):
            break

        if (case[idx + 1] == case[idx]):
            res[idx] = event[idx + 1]

        idx += 1
    return res


data['next_event'] = calculator_nb(data['case'].values, data['event'].values)


#Previous event
@jit(parallel=True)
def calculator_nb(case, event):
    res = np.empty(len(case), dtype=object)
    idx = 0
    for _ in case:
        if (idx + 1 >= len(case)):
            break

        if (case[idx + 1] == case[idx]):
            res[idx + 1] = event[idx]

        idx += 1
    return res


data['prev_event'] = calculator_nb(data['case'].values, data['event'].values)

#Removing null values
data['next_event'] = data['next_event'].fillna(value='None')
data['prev_event'] = data['prev_event'].fillna(value='None')

#unix time
pd.set_option('display.float_format', lambda x: '%.3f' % x)

data['startTime'] = pd.to_datetime(data['startTime'], dayfirst=True)
unixTransform = lambda x: time.mktime(x.timetuple())
data["UNIX_starttime"] = data["startTime"].apply(unixTransform).astype(int)

data['completeTime'] = pd.to_datetime(data['completeTime'], dayfirst=True)
unixTransform = lambda x: time.mktime(x.timetuple())
data["UNIX_completeTime"] = data["completeTime"].apply(unixTransform).astype(
    int)

#data['REG_DATE'] = pd.to_datetime(data['REG_DATE'], dayfirst=True)
#unixTransform = lambda x: time.mktime(x.timetuple())
#data["UNIX_REG_DATE"] = data["REG_DATE"].apply(unixTransform).astype(int)

#print(data)
#Day of the week
data['weekday'] = data['startTime'].dt.dayofweek
#encoding of categorical data
ordinal_encoder = OrdinalEncoder()
label_encoder = LabelEncoder()
data['enc_event'] = ordinal_encoder.fit_transform(data[['event']]).astype(int)
#ensure we have acces to orignal indexing to keep track of the order of events in a process
data['original index'] = data.index

#sorting on time
data.sort_values(by="UNIX_starttime", ignore_index=True)

#separation
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, shuffle=False)

#removing overlap - if case is in both datasets, remove

train_cases = train['case'].unique().tolist()
test_cases = test['case'].unique().tolist()

intersect_list = list(set(train_cases).intersection(test_cases))
#only removes first value in intersect list (needs modification for multiple overlaping values)

#train = train[train['case'] != intersect_list[0]]
#test = test[test['case'] != intersect_list[0]]

#works for more values
org_train = train.copy()
org_test = test.copy()
df_ordinal_encoder = LabelEncoder()
train = train.apply(df_ordinal_encoder.fit_transform)
test = test.apply(df_ordinal_encoder.fit_transform)

train = train[train['case'].isin(intersect_list) == False]
X_train_time = train.drop(columns='duration')
Y_train_time = train["duration"]
X_train_event = train.drop(columns=["next_event"])
Y_train_event = train["next_event"]

test = test[test['case'].isin(intersect_list) == False]
X_test_time = test.drop(columns='duration')
Y_test_time = test["duration"]
X_test_event = test.drop(columns=['next_event'])
Y_test_event = test["next_event"]

# Random forest event
# DT = DecisionTreeClassifier()


def calc_feature_selection():
    select = SelectKBest(k=10)  # takes best 10 arguments
    z = select.fit_transform(X_train_time, Y_train_time)
    filter = select.get_support()
    print(np.extract(filter, select.scores_))
    print(np.extract(filter, X_train_time.columns))
    #['event' 'penalty_AVBP' 'penalty_AVGP' 'eventid' 'activity' 'docid' 'subprocess' 'success' 'next_event' 'enc_event'] for time

    select = SelectKBest(k=10)  # takes best 10 arguments
    z = select.fit_transform(X_train_event, Y_train_event)
    filter = select.get_support()
    print(np.extract(filter, select.scores_))
    print(np.extract(filter, X_train_event.columns))
    # ['event' 'selected_random' 'note' 'eventid' 'activity' 'subprocess' 'org:resource' 'duration' 'prev_event' 'enc_event'] for event


def tune_rf():
    param_grid = {
        'bootstrap': [True],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, None],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
    }
    RF = RandomForestClassifier(
        # n_estimators=300,
        #                         min_samples_split=10,
        #                         min_samples_leaf=2,
        #                         max_features='sqrt',
        #                         max_depth=50,
        #                         bootstrap=True
    )
    RF = GridSearchCV(
        estimator=RF,
        param_grid=param_grid,
        scoring='accuracy',
        #   n_iter=1,
        #   cv=5,
        verbose=2,
        # random_state=42,
        n_jobs=6)
    dataset_col = [
        'event',
        'selected_random',
        'note',
        'eventid',
        'activity',
        'subprocess',
        'org:resource',
        'duration',
        'prev_event',
    ]

    RF_fit = RF.fit(X_train_event[:10000].filter(items=dataset_col),
                    Y_train_event[:10000])
    print(RF.best_params_)
    RF_pred = RF_fit.predict(X_test_event[:10000].filter(items=dataset_col))
    org_test["event_RF"] = RF_pred
    print("Accuracy for Random Forest: ",
          accuracy_score(Y_test_event[:10000], RF_pred[:10000]))


def calc_random_forest():
    # Create the random grid

    RF = RandomForestClassifier(n_jobs=6,
                                verbose=2,
                                n_estimators=100,
                                min_samples_split=5,
                                min_samples_leaf=1,
                                max_features='auto',
                                max_depth=80,
                                bootstrap=True)

    dataset_col = [
        'event',
        'selected_random',
        'note',
        'eventid',
        'activity',
        'subprocess',
        'org:resource',
        'duration',
        'prev_event',
    ]

    RF_fit = RF.fit(X_train_event.filter(items=dataset_col), Y_train_event)
    # print(RF_fit.best)
    RF_pred = RF_fit.predict(X_test_event.filter(items=dataset_col))
    org_test["event_RF"] = RF_pred
    print("Accuracy for Random Forest: ",
          accuracy_score(Y_test_event, RF_pred))


def calc_LSTM():
    listVal = X_train_time
    columnNames = [
        'event', 'penalty_AVBP', 'penalty_AVGP', 'eventid', 'activity',
        'docid', 'subprocess', 'success', 'next_event', 'enc_event'
    ]
    listValSelected = listVal[columnNames]
    listValSelected_prediction = X_test_time[columnNames]
    listValSelected_prediction = listValSelected_prediction.values
    listValDuration_prediction = org_test['duration']
    listValDuration_prediction = listValDuration_prediction.values
    listValDuration = org_train['duration'].values

    listValSelected = listValSelected.values
    n_steps = len(listValSelected[0])
    # split into samples
    n_features = 1
    X = listValSelected.reshape(
        (listValSelected.shape[0], listValSelected.shape[1], n_features))
    y = listValDuration

    # define model
    model = Sequential()
    model.add(
        LSTM(
            100,
            input_shape=(n_steps, n_features),
            #  stateful=True,
            return_sequences=False))
    model.add(Dropout(0.2))
    # model.add(LSTM(units=100))
    # model.add(Dropout(0.2))
    # model.add(Dense(units=1))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='mse')

    model.fit(X,
              y,
              batch_size=400,
              epochs=5,
              verbose=1,
              workers=-1,
              use_multiprocessing=True)

    # demonstrate prediction
    x_input = listValSelected_prediction

    x_input = x_input.reshape((x_input.shape[0], n_steps, n_features))
    yhat = model.predict(x_input, verbose=1, use_multiprocessing=True)
    print(
        mean_absolute_error(listValDuration_prediction,
                            yhat.flatten()[:len(listValDuration_prediction)]))
    print(yhat)
    # return yhat.flatten()


# calc_feature_selection()
# calc_random_forest()
# calc_LSTM()
# tune_rf()
# calc_event()
# linear_regression()