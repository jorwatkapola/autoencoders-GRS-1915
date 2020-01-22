from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import pickle
from scipy.stats import zscore
import datetime
import pytz

np.random.seed(seed=11)

with open('series_34697_1000.pkl', 'rb') as f:
    segments = pickle.load(f)
    
segments = zscore(segments)  # standardize


deep_model = Sequential(name="LSTM-autoencoder")
deep_model.add(LSTM(400, activation='tanh', input_shape=(1000,1), return_sequences=True, dropout=0.4))
deep_model.add(LSTM(200, activation='tanh', return_sequences=True, dropout=0.4))
deep_model.add(LSTM(100, activation='tanh', return_sequences=False, dropout=0.4))
deep_model.add(Dense(100, activation=None))
deep_model.add(RepeatVector(1000))
deep_model.add(LSTM(100, activation='tanh', return_sequences=True, dropout=0.4))
deep_model.add(LSTM(200, activation='tanh', return_sequences=True, dropout=0.4))
deep_model.add(LSTM(400, activation='tanh', return_sequences=True, dropout=0.4))
deep_model.add(TimeDistributed(Dense(1)))
deep_model.compile(optimizer='adam', loss='mse')

training_time_stamp = datetime.datetime.now(tz=pytz.timezone('Europe/London')).strftime("%Y-%m-%d_%H-%M-%S")

CB = EarlyStopping(monitor='val_loss', min_delta=1e-1, patience=5, verbose=1, mode='auto')
MC = ModelCheckpoint('lstm_autoencoder_{}.h5'.format(training_time_stamp), monitor='val_loss', mode="auto", save_best_only=True, verbose=1)
history = deep_model.fit(segments, segments, epochs=100, verbose=1, callbacks = [MC, CB], validation_split=0.05)

np.savetxt("loss_history-{}.txt".format(training_time_stamp), [np.asarray(history.history["loss"]), np.asarray(history.history["val_loss"])], delimiter=",")