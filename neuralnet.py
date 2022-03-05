import numpy as np  # import auxiliary library, typical idiom
import pandas as pd  # import the Pandas library, typical idiom
from numba import jit
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from tqdm.keras import TqdmCallback

df = pd.read_csv('export.csv')

df_next_event = df.copy()

#A function for determining the true next event for each event
@jit(parallel = True)
def calculator_nb(case, event):
    res = np.empty(len(case), dtype=object)
    idx = 0
    for _ in case:
        if (idx+1 >= len(case)):
            break
       
        if (case[idx + 1] == case[idx]):
            res[idx] = event[idx + 1]

        idx+=1
    return res

df_next_event['next_event'] = calculator_nb(df_next_event['case'].values, df_next_event['event'].values)

df = df_next_event

goodshit = ['case','event','number_parcels','payment_actual0','area','cross_compliance','penalty_amount0','next_event']
df_reduced = df.copy()

for col in df.columns:
    if (goodshit.count(col) == 0):
        df_reduced = df_reduced.drop(col, 1)

def normalize(col_name):
    col_as_array = df_reduced[col_name].to_numpy()
    col_as_array = np.where(col_as_array == 0, 0.5, col_as_array)
    col_as_array_norm = np.log10(col_as_array)
    mean = col_as_array_norm.mean()
    stdev = col_as_array_norm.std()
    epsilon = 0.01
    return (col_as_array_norm - mean) / (stdev + epsilon)

df_reduced = df_reduced.fillna(value="None")

df_reduced, validate, test = np.split(df_reduced.sample(frac=1), [int(.6*len(df_reduced)), int(.8*len(df_reduced))])

event = df_reduced['event'].to_numpy()
next_event = df_reduced['next_event'].to_numpy()

ordinal_encoder = OrdinalEncoder()
label_encoder = LabelEncoder()
event = event.reshape(-1, 1)
event_encoded = ordinal_encoder.fit_transform(event)
next_event_encoded = label_encoder.fit_transform(next_event)
next_event_encoded = next_event_encoded.reshape(-1, 1)

number_parcels = normalize('number_parcels')
payment_actual0 = normalize('payment_actual0')
area = normalize('area')
cross_compliance = normalize('cross_compliance')
penalty_amount0 = normalize('penalty_amount0')

training = []
for i in range(len(event_encoded)):
    current = event_encoded[i]
    current = np.append(current,number_parcels[i])
    current = np.append(current,payment_actual0[i])
    current = np.append(current,area[i])
    current = np.append(current,cross_compliance[i])
    current = np.append(current,penalty_amount0[i])
    training.append(current)

labels = []
for i in range(len(next_event_encoded)):
    current = next_event_encoded[i]
    current = np.append(current,[0,0,0,0,0])
    labels.append(current)

labels = labels[:200000]
training = training[:200000]

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(6,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(43, activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(training,labels,epochs=1,verbose=0,callbacks=[TqdmCallback(verbose=2)])