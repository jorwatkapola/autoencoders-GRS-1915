import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.backend import mean
from tensorflow.keras.backend import square


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten

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

with open('../../../data_GRS1915/191660_len256_stride4_4sec_counts.pkl', 'rb') as f:
    segments = pickle.load(f)
with open('../../../data_GRS1915/191660_len256_stride4_4sec_errors.pkl', 'rb') as f:
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


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self,
                   latent_dim=16,
                   intermediate_dim=1024,
                   name='encoder',
                   **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.CuDNNLSTM(intermediate_dim, return_sequences=False)
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self,
                   original_dim,
                   intermediate_dim=1024,
                   name='decoder',
                   **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.repeat = layers.RepeatVector(original_dim)
        self.dense_proj = layers.CuDNNLSTM(intermediate_dim, return_sequences=True)
        self.dense_output = layers.TimeDistributed(layers.Dense(1))

    def call(self, inputs):
        inputs_repeated = self.repeat(inputs)
        x = self.dense_proj(inputs_repeated)
        return self.dense_output(x)


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
                   original_dim,
                   intermediate_dim=1024,
                   latent_dim=16,
                   name='autoencoder',
                   **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                               intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = - 0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed

    
# Training and Validation generators in a 95/5 split
training_generator = DataGenerator(segments[:int(np.floor(len(segments)*0.95))], errors[:int(np.floor(len(errors)*0.95))], batch_size=512)
validation_generator = DataGenerator(segments[int(np.floor(len(segments)*0.95)):], errors[int(np.floor(len(errors)*0.95)):], batch_size=512)    

y_in = Input(shape=(256,1))
y_err = Input(shape=(256,1))
vae = VariationalAutoEncoder(256, 1024, 16)

optimizer = tf.keras.optimizers.Adam(clipvalue=0.5)

vae.compile(optimizer, loss=chi2(y_err))



# vae.fit(x_train, x_train, epochs=3, batch_size=64)

training_time_stamp = datetime.datetime.now(tz=pytz.timezone('Europe/London')).strftime("%Y-%m-%d_%H-%M-%S")

CB = EarlyStopping(monitor='val_loss', min_delta=5e-5, patience=100, verbose=1, mode='auto')
MC = ModelCheckpoint('../../model_weights/model_{}.h5'.format(training_time_stamp), monitor='val_loss', mode="auto", save_best_only=True, verbose=1)
history = vae.fit_generator(training_generator, epochs=8000, verbose=2, callbacks = [MC, CB], validation_data=validation_generator)


np.savetxt("training_history/loss_history-{}.txt".format(training_time_stamp), [np.asarray(history.history["loss"]), np.asarray(history.history["val_loss"])], delimiter=",")
