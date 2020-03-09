from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import CuDNNGRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD

import numpy as np
import pickle
from scipy.stats import zscore
import datetime
import pytz

np.random.seed(seed=11)

with open('series_34697_1000.pkl', 'rb') as f:
    segments = pickle.load(f)
    
segments = zscore(segments).astype(np.float32)  # standardize



deep_model = Sequential(name="LSTM-autoencoder")
deep_model.add(CuDNNGRU(200, input_shape=(1000,1), return_sequences=True))
deep_model.add(CuDNNGRU(100, return_sequences=False))
deep_model.add(Dense(20, activation=None))
deep_model.add(RepeatVector(1000))
deep_model.add(CuDNNGRU(100, return_sequences=True))
deep_model.add(CuDNNGRU(200, return_sequences=True))
deep_model.add(TimeDistributed(Dense(1)))
deep_model.compile(optimizer=SGD(lr=5e-3, momentum=0.9, clipnorm=1.0), loss='mse')

#deep_model.load_weights("lstm_autoencoder_2019-12-21_00-01-05.h5")

training_time_stamp = datetime.datetime.now(tz=pytz.timezone('Europe/London')).strftime("%Y-%m-%d_%H-%M-%S")

CB = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1, mode='auto')
MC = ModelCheckpoint('model_weights/lstm_autoencoder_{}.h5'.format(training_time_stamp), monitor='val_loss', mode="auto", save_best_only=True, verbose=1)
history = deep_model.fit(segments, segments, epochs=8000, verbose=2, callbacks = [MC], validation_split=0.05, batch_size=384)

np.savetxt("loss_history-{}.txt".format(training_time_stamp), [np.asarray(history.history["loss"]), np.asarray(history.history["val_loss"])], delimiter=",")
