from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import CuDNNGRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam

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
deep_model.add(CuDNNGRU(100, input_shape=(1000,1), return_sequences=True))
deep_model.add(CuDNNGRU(100, return_sequences=True))
deep_model.add(CuDNNGRU(100, return_sequences=True))
deep_model.add(CuDNNGRU(100, return_sequences=False))
deep_model.add(Dense(64, activation=None))
deep_model.add(RepeatVector(1000))
deep_model.add(CuDNNGRU(100, return_sequences=True))
deep_model.add(CuDNNGRU(100, return_sequences=True))
deep_model.add(CuDNNGRU(100, return_sequences=True))
deep_model.add(CuDNNGRU(100, return_sequences=True))
deep_model.add(TimeDistributed(Dense(1)))
deep_model.compile(optimizer=Adam(lr=3e-4, clipnorm=1.0), loss='mse')

deep_model.load_weights("lstm_autoencoder_2019-12-18_07-39-25.h5")

training_time_stamp = datetime.datetime.now(tz=pytz.timezone('Europe/London')).strftime("%Y-%m-%d_%H-%M-%S")

CB = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=15, verbose=1, mode='auto')
MC = ModelCheckpoint('lstm_autoencoder_{}.h5'.format(training_time_stamp), monitor='val_loss', mode="auto", save_best_only=True, verbose=1)
history = deep_model.fit(segments, segments, epochs=250, verbose=2, callbacks = [MC, CB], validation_split=0.05, batch_size=256)

np.savetxt("loss_history-{}.txt".format(training_time_stamp), [np.asarray(history.history["loss"]), np.asarray(history.history["val_loss"])], delimiter=",")
