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
from tensorflow import keras
from sklearn.linear_model import HuberRegressor
import tracemalloc

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
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from xgboost import *
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

tracemalloc.start()


def time_metrics(y_test, y_pred, model="..."):

    print("\n")
    print(f"Error metrics for the {model} time model")
    print("\n")
    print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:',
          np.sqrt(mean_squared_error(y_test, y_pred)))
    print('$R_2$ score:', r2_score(y_test, y_pred))
    print("\n")


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

data = data[:1000]

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
org_train = train.copy()[:1000]
org_test = test.copy()[:1000]
df_ordinal_encoder = LabelEncoder()
train = train.apply(df_ordinal_encoder.fit_transform)
test = test.apply(df_ordinal_encoder.fit_transform)

train = train[train['case'].isin(intersect_list) == False][:1000]
X_train_time = train.drop(columns='duration')
Y_train_time = train["duration"]
X_train_event = train.drop(columns=["next_event"])
Y_train_event = train["next_event"]

test = test[test['case'].isin(intersect_list) == False][:1000]
X_test_time = test.drop(columns='duration')
Y_test_time = test["duration"]
X_test_event = test.drop(columns=['next_event'])
Y_test_event = test["next_event"]

# Random forest event

# Naive event
data_baseline = org_train[[
    "case", "event", "startTime", "completeTime", "next_event", "enc_event",
    "original index", "UNIX_starttime", "UNIX_completeTime", "duration"
]].copy()
test_baseline = org_test[[
    "case", "event", "startTime", "completeTime", "next_event", "enc_event",
    "original index", "UNIX_starttime", "UNIX_completeTime", "duration"
]].copy()


# Naive Bayes
def naive_baseline():
    @jit(parallel=True)
    def calculator_pos(case):
        res = np.empty(len(case), dtype=object)
        idx = 0
        count = 1
        for _ in case:
            if (idx + 1 >= len(case)):
                break
            if (case[idx] == case[idx - 1]):
                count += 1
                res[idx] = count
            else:
                count = 1
                res[idx] = count
            idx += 1
        res[-1] = count + 1
        return res

    data_baseline["pos"] = calculator_pos(data_baseline['case'].to_numpy())
    #select most occuring event for each position
    events_count = data_baseline.groupby("pos")['enc_event'].agg(
        lambda x: pd.Series.mode(x)[0]).to_frame()
    events_count = events_count.rename(columns={"pos": "enc_event"})
    #Next event for each position (most occuring event in the next position, e.g. 1-25, 2-26, so for 1 we predict 26)
    events_count['next_event2'] = events_count['enc_event'].shift(-1)
    events_count["next_event2"].iloc[-1] = 0
    #map the model results to the original dataframe
    data_baseline["next_event2"] = data_baseline["pos"].map(
        events_count["next_event2"])
    #Cleaning, set last events for each case to -1, so that we don't use them in final result and error estimation.
    data_baseline["index"] = data_baseline.index
    #find last position for each case
    last_pos_per_case = data_baseline.groupby("case")[["index", "case",
                                                       "pos"]].agg(max)
    #use index of the last event per case to assign -1 to the last event
    last_pos_per_case.set_index('index', inplace=True)
    #last_pos_per_case = last_pos_per_case["next_event2"]
    data_baseline.loc[last_pos_per_case.index, "next_event2"] = -1

    #Use the naive event model on test dataset
    test_baseline["pos"] = calculator_pos(test_baseline['case'].to_numpy())
    test_baseline["next_event2"] = test_baseline["pos"].map(
        events_count["next_event2"])
    test_baseline["index"] = test_baseline.index
    last_pos_per_case = test_baseline.groupby("case")[["index", "case",
                                                       "pos"]].agg(max)
    last_pos_per_case.set_index('index', inplace=True)
    test_baseline.loc[last_pos_per_case.index, "next_event2"] = -1
    y_true_event = test_baseline["enc_event"].to_numpy().astype(int)
    y_pred_event = test_baseline["next_event2"].to_numpy().astype(int)

    # Naive time predictor
    #find difference between completeTime of current event and the next event
    data_baseline["duration_start-start"] = pd.to_numeric(
        data_baseline["UNIX_starttime"].diff(), downcast='signed')
    #shift it up, so the difference corresponds to the current event
    data_baseline["duration_start-start"] = pd.to_numeric(
        data_baseline["duration_start-start"].shift(-1))
    #set time duration between cases to NaT (last event per case, last position), since we only want time duration per case
    data_baseline.loc[data_baseline[data_baseline["next_event2"] == -1].index,
                      "duration_start-start"] = None
    #find average time duration between startTime of the events at current position and at the next position
    predicted_duration = data_baseline.groupby(
        "pos")['duration_start-start'].agg('mean')
    data_baseline["predicted_duration"] = data_baseline['pos'].map(
        predicted_duration)
    #add predicted duration to completeTime to predict the startTime of the event in the next position
    data_baseline["predicted_time"] = data_baseline[
        "UNIX_starttime"] + data_baseline["predicted_duration"]
    #naive predictor for test dataset
    test_baseline["predicted_duration"] = test_baseline['pos'].map(
        predicted_duration)
    test_baseline["predicted_time"] = test_baseline[
        "UNIX_starttime"] + test_baseline["predicted_duration"]
    y_true_time = test_baseline["UNIX_starttime"].to_numpy().astype(int)
    y_pred_time = test_baseline["predicted_time"].to_numpy().astype(int)
    #Error measurement
    event_metrics(y_true_event, y_pred_event, model="Naive_event")
    time_metrics(y_true_time, y_pred_time, model="Naive_time")
    #add columns
    org_test["event_naive"] = ordinal_encoder.inverse_transform(
        test_baseline[["next_event2"]])
    org_test.loc[test_baseline[test_baseline["next_event2"] == -1].index,
                 "event_naive"] = None
    org_test["duration_naive"] = test_baseline["predicted_duration"]
    org_test["startTime_naive"] = test_baseline["predicted_time"]


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
        'subprocess',
        'org:resource',
        'duration',
        'prev_event',
    ]

    RF_fit = RF.fit(X_train_event.filter(items=dataset_col), Y_train_event)
    print(RF.best_params_)
    RF_pred = RF_fit.predict(X_test_event.filter(items=dataset_col))
    org_test["event_RF"] = RF_pred
    print("Accuracy for Random Forest: ",
          accuracy_score(Y_test_event, RF_pred))


import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def event_metrics(y_test, y_pred, avg="weighted", model="..."):

    prec_score = precision_score(y_test, y_pred, average=avg, zero_division=0)
    rec_score = recall_score(y_test, y_pred, average=avg, zero_division=0)
    F1_score = f1_score(y_test, y_pred, average=avg, zero_division=0)
    acc_score = accuracy_score(y_test, y_pred)

    print("\n")
    print(f"Error metrics for the {model} event model")
    print("\n")
    print(f'The accuracy of the model is {acc_score}.')
    print(f'The precision of the model is {prec_score}, using {avg} average.')
    print(f'The recall of the model is {rec_score}, using {avg} average.')
    print(f'The f1-score of the model is {F1_score}, using {avg} average.')
    print("\n")

    return acc_score, prec_score, rec_score, F1_score


def evaluate_model(model, X, y):
    # define the model evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


def calc_xgboost_rf():
    encoder = LabelEncoder()
    encoder.fit(org_train["event"])
    model = XGBRFClassifier(n_estimators=100,
                            subsample=0.9,
                            colsample_bynode=0.2,
                            use_label_encoder=True)
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    dataset_col = [
        'event', 'selected_random', 'subprocess', 'org:resource', 'duration',
        'prev_event', 'weekday'
    ]
    X = X_train_event.filter(items=dataset_col)
    y = Y_train_event
    model.fit(X, y, verbose=True)
    prediction = model.predict(X_test_event.filter(items=dataset_col))
    org_test["rf_event"] = encoder.inverse_transform(prediction)
    event_metrics(Y_test_event, prediction, model="RF")


def calc_random_forest():
    # Create the random grid

    RF = RandomForestClassifier(n_estimators=100,
                                min_samples_split=10,
                                min_samples_leaf=2,
                                max_features='sqrt',
                                max_depth=50,
                                bootstrap=True)

    dataset_col = [
        'event', 'selected_random', 'subprocess', 'org:resource', 'duration',
        'prev_event', 'weekday'
    ]

    RF_fit = RF.fit(X_train_event.filter(items=dataset_col), Y_train_event)
    RF_pred = RF_fit.predict(X_test_event.filter(items=dataset_col))
    org_test["event_RF"] = RF_pred
    event_metrics(Y_test_event, RF_pred, model="Random Forest")


def calc_LSTM():
    listVal = X_train_time
    columnNames = [
        'event', 'penalty_AVBP', 'penalty_AVGP', 'eventid', 'activity',
        'docid', 'subprocess', 'success', 'next_event', 'weekday'
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
    model.add(Activation('softmax'))
    print('The CPU usage is: ', psutil.cpu_percent(4))
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
    prediction = yhat.flatten()[:len(listValDuration_prediction
                                     )] + org_test["UNIX_starttime"]
    time_metrics(listValDuration_prediction,
                 yhat.flatten()[:len(listValDuration_prediction)])
    org_test["lstm_duration_pred"] = prediction


# naive_baseline()
# naive_time()
# calc_feature_selection()
# calc_random_forest()
# calc_LSTM()
# tune_rf()
calc_xgboost_rf()


def normalize(df_name, col_name):
    col_as_array = df_name[col_name].to_numpy()
    col_as_array = np.where(col_as_array == 0, 0.01, col_as_array)
    col_as_array_norm = np.log10(col_as_array)
    mean = col_as_array_norm.mean()
    stdev = col_as_array_norm.std()
    epsilon = 0.01
    return (col_as_array_norm - mean) / (stdev + epsilon)


def prepfeatures(df_name):
    event = df_name['event'].to_numpy()
    event = event.reshape(-1, 1)
    event = ordinal_encoder.fit_transform(event)

    duration = normalize(df_name, 'duration')
    weekday = df_name['weekday'].to_numpy()

    prev_event = df_name['prev_event'].to_numpy()
    prev_event = prev_event.reshape(-1, 1)
    prev_event = ordinal_encoder.fit_transform(prev_event)

    features = []
    for i in range(len(event)):
        current = event[i]
        current = np.append(current, duration[i])
        current = np.append(current, weekday[i])
        current = np.append(current, prev_event[i])
        features.append(current)

    return np.array(features)


def preplabels(df_name):
    labels = df_name['next_event'].to_numpy()
    labels = label_encoder.fit_transform(labels)
    labels = labels.reshape(-1, 1)

    return np.array(labels)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(4, )),
    keras.layers.Dense(20, activation='softplus'),
    keras.layers.Dropout(1 / 10),
    keras.layers.Dense(30, activation='softplus'),
    keras.layers.Dropout(1 / 15),
    keras.layers.Dense(42, activation='softplus')
])

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

losses = []
accuracies = []


def crossvalidation(k):
    quantile = int(np.floor(len(data) / 5))
    for i in range(1, k - 1):
        train = data[:(quantile * i)]
        train = train.sample(frac=1)
        test = data[(quantile * i):(quantile * (i + 1))]

        features = prepfeatures(train)
        labels = preplabels(train)

        print("Training on 0:", (quantile * i), "; Testing on ",
              (quantile * i), ":", (quantile * (i + 1)))
        model.fit(features, labels, epochs=3, verbose=1)

        features_test = prepfeatures(test)
        labels_test = preplabels(test)

        eval = model.evaluate(features_test, labels_test)
        losses.append(eval[0])
        accuracies.append(eval[1])

    return losses, accuracies


losses, accuracies = crossvalidation(5)

np.array(accuracies).mean()
features_data = prepfeatures(org_test)
prediction = model.predict(features_data)
predicted_events = []
for i in range(len(prediction)):
    predicted_events.append(np.argmax(prediction[i]))

predicted_events = label_encoder.inverse_transform(predicted_events)

#event_metrics(data["next_event"], predicted_events)

org_test['neuralnet_event_prediction'] = predicted_events


def prepfeatures_regression(df_name):
    event = df_name['event'].to_numpy()
    event = event.reshape(-1, 1)
    event = ordinal_encoder.fit_transform(event)

    next_event = df_name['next_event'].to_numpy()
    next_event = next_event.reshape(-1, 1)
    next_event = ordinal_encoder.fit_transform(next_event)

    year = df_name['year'].to_numpy()
    year = year.reshape(-1, 1)
    year = ordinal_encoder.fit_transform(year)

    penalty_AVBP = df_name['penalty_AVBP'].to_numpy()
    penalty_AVBP = penalty_AVBP.reshape(-1, 1)
    penalty_AVBP = ordinal_encoder.fit_transform(penalty_AVBP)

    penalty_AVGP = df_name['penalty_AVGP'].to_numpy()
    penalty_AVGP = penalty_AVGP.reshape(-1, 1)
    penalty_AVGP = ordinal_encoder.fit_transform(penalty_AVGP)

    success = df_name['success'].to_numpy()
    success = success.reshape(-1, 1)
    success = ordinal_encoder.fit_transform(success)

    eventid = df_name['eventid'].to_numpy()
    eventid = success.reshape(-1, 1)

    docid = df_name['docid'].to_numpy()
    docid = success.reshape(-1, 1)

    subprocess = df_name['subprocess'].to_numpy()
    subprocess = subprocess.reshape(-1, 1)
    subprocess = ordinal_encoder.fit_transform(subprocess)

    weekday = df_name['weekday'].to_numpy()
    weekday = weekday.reshape(-1, 1)

    X = []
    for i in range(len(event)):
        current = event[i]
        current = np.append(current, year[i])
        current = np.append(current, penalty_AVBP[i])
        current = np.append(current, penalty_AVGP[i])
        current = np.append(current, success[i])
        current = np.append(current, eventid[i])
        current = np.append(current, docid[i])
        current = np.append(current, next_event[i])
        current = np.append(current, subprocess[i])
        current = np.append(current, weekday[i])
        X.append(current)

    return np.array(X, dtype=float)


def preplabels_regression(df_name):
    duration = df_name['duration'].to_numpy()
    return np.array(duration, dtype=float)


X = prepfeatures_regression(org_train)
y = preplabels_regression(org_train)
huber = HuberRegressor().fit(X, y)

X_test = prepfeatures_regression(org_test)

org_test['regression_duration'] = huber.predict(X_test)
org_test['error'] = np.absolute(org_test['duration'] -
                                org_test['regression_duration'])
org_test['error'].mean()

time_metrics(org_test["duration"], org_test['regression_duration'])

# time_metrics(org_test['duration'], data['regression_duration'])
X_for_prediction = prepfeatures_regression(org_test)
org_test['regression_time_prediction'] = huber.predict(X_for_prediction)
org_test['regression_time_prediction'] = org_test[
    'regression_time_prediction'] + org_test['UNIX_starttime']
org_test['regression_time_predicition'] = org_test[
    'regression_time_prediction'].apply(datetime.fromtimestamp)

current, peak = tracemalloc.get_traced_memory()
print(
    f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

org_test.to_csv("exported.csv")

tracemalloc.stop()