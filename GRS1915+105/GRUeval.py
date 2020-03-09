from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import CuDNNGRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.backend import variable
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
deep_model.add(CuDNNGRU(100, input_shape=(1000,1), return_sequences=True))
deep_model.add(CuDNNGRU(50, return_sequences=True))
deep_model.add(CuDNNGRU(25, return_sequences=False))
deep_model.add(Dense(25, activation=None))
deep_model.add(RepeatVector(1000))
deep_model.add(CuDNNGRU(25, return_sequences=True))
deep_model.add(CuDNNGRU(50, return_sequences=True))
deep_model.add(CuDNNGRU(100, return_sequences=True))
deep_model.add(TimeDistributed(Dense(1)))


deep_model.load_weights('lstm_autoencoder_2019-12-16_11-56-02.h5')

reco = deep_model(Input(np.reshape(segments[3], (1,1000,1))))

np.savetxt("reco.txt", reco, delimiter=",")