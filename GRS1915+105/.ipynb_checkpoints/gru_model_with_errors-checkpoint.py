from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import CuDNNGRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten
from tensorflow.keras.backend import mean
from tensorflow.keras.backend import square
from tensorflow.keras.utils import Sequence
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
import numpy as np
import pickle
from scipy.stats import zscore
import datetime
import pytz

np.random.seed(seed=11)

with open('../data/95921_len512_s40_counts.pkl', 'rb') as f:
    segments = pickle.load(f)
with open('../data/95921_len512_s40_errors.pkl', 'rb') as f:
    errors = pickle.load(f)
    
errors = ((errors)/np.std(segments)).astype(np.float32)
segments = zscore(segments, axis=None).astype(np.float32)  # standardize

def chi2(y_err):
    def MSE_scaled(y_in, y_out,):
        return mean(square(y_in-y_out)/square(y_err))
    return MSE_scaled


class DataGenerator(Sequence):
    """
    Generates data for Keras
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    https://stackoverflow.com/questions/53105294/implementing-a-batch-dependent-loss-in-keras
    """
    def __init__(self, y_in, y_err, batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.y_in = y_in
        self.y_err = y_err        
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y_in) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        y_in = self.y_in[indexes]
        y_err = self.y_err[indexes]
        return [y_in, y_err], y_in
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.y_in))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


# Training and Validation generators in a 95/5 split
training_generator = DataGenerator(segments[:int(np.floor(len(segments)*0.95))], errors[:int(np.floor(len(errors)*0.95))], batch_size=128)
validation_generator = DataGenerator(segments[int(np.floor(len(segments)*0.95)):], errors[int(np.floor(len(errors)*0.95)):], batch_size=128)


y_in = Input(shape=(512,1))
y_err = Input(shape=(512,1))
# h_enc = Conv1D(32, 2, activation='relu')(y_in)
# h_enc = Conv1D(32, 8, activation='relu')(h_enc)
# h_enc = CuDNNGRU(512, return_sequences=True)(y_in)
# h_enc = CuDNNGRU(256, return_sequences=True)(y_in)
h_enc = CuDNNGRU(2048, return_sequences=False)(y_in)
# h_enc = BatchNormalization()(h_enc)
h_enc = Dense(16, activation=None, name='bottleneck')(h_enc)
# h_enc = BatchNormalization()(h_enc)
h_dec = RepeatVector(512)(h_enc)
h_dec = CuDNNGRU(2048, return_sequences=True)(h_dec)
# h_dec = CuDNNGRU(256, return_sequences=True)(h_dec)
h_dec = TimeDistributed(Dense(1))(h_dec)
model = Model(inputs=[y_in, y_err], outputs=h_dec)
model.compile(optimizer=Adam(clipvalue=0.5), loss=chi2(y_err))

# model.load_weights("model_weights/lstm_autoencoder_2020-01-09_18-20-21.h5")

training_time_stamp = datetime.datetime.now(tz=pytz.timezone('Europe/London')).strftime("%Y-%m-%d_%H-%M-%S")

CB = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=100, verbose=1, mode='auto')
MC = ModelCheckpoint('../model_weights/model_{}.h5'.format(training_time_stamp), monitor='val_loss', mode="auto", save_best_only=True, verbose=1)
history = model.fit_generator(training_generator, epochs=8000, verbose=2, callbacks = [MC, CB], validation_data=validation_generator)


np.savetxt("training_history/loss_history-{}.txt".format(training_time_stamp), [np.asarray(history.history["loss"]), np.asarray(history.history["val_loss"])], delimiter=",")
