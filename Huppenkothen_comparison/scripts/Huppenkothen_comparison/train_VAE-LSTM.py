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
from tensorflow.keras.optimizers import SGD
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


data_dir = "../../../data_GRS1915"


with open('{}/segments_1024s_256stride_0125cad_segmented_to64_train.pkl'.format(data_dir), 'rb') as f:
    train_data = pickle.load(f)
with open('{}/segments_1024s_256stride_0125cad_segmented_to64_valid.pkl'.format(data_dir), 'rb') as f:
    valid_data = pickle.load(f)
    
cadence = np.min(np.diff([seg[2][0] for seg in train_data]))

# get rid of the meta-data and make numpy arrays with required dimensions
train_data_counts = [seg[2][1] for seg in train_data]
valid_data_counts = [seg[2][1] for seg in valid_data]

#divide by cadence to turn counts to count rates
train_data_counts = np.vstack(train_data_counts) /cadence
valid_data_counts = np.vstack(valid_data_counts) /cadence

train_data_counts = np.expand_dims(train_data_counts, axis=-1)
valid_data_counts = np.expand_dims(valid_data_counts, axis=-1)

train_data_errors = [seg[2][2] for seg in train_data]
valid_data_errors = [seg[2][2] for seg in valid_data]

train_data_errors = np.vstack(train_data_errors)
valid_data_errors = np.vstack(valid_data_errors)

#error values must be non-zero. replace zeros with a small value
min_nonzero_train = np.min(train_data_errors[train_data_errors!=0])/10
min_nonzero_valid = np.min(valid_data_errors[valid_data_errors!=0])/10

train_data_errors[train_data_errors==0] = min_nonzero_train
valid_data_errors[valid_data_errors==0] = min_nonzero_valid

train_data_errors = np.expand_dims(train_data_errors, axis=-1)
valid_data_errors = np.expand_dims(valid_data_errors, axis=-1)


# standardize data per segment
#should we use std of all training data here?
train_data_errors = ((train_data_errors)/np.expand_dims(np.std(train_data_counts, axis=1), axis=1)).astype(np.float32)
valid_data_errors = ((valid_data_errors)/np.expand_dims(np.std(valid_data_counts, axis=1), axis=1)).astype(np.float32)

train_data_counts = zscore(train_data_counts, axis=1).astype(np.float32)  
valid_data_counts = zscore(valid_data_counts, axis=1).astype(np.float32)  




def chi2(y_err):
    def MSE_scaled(y_in, y_out,):
        return mean(square((y_in-y_out)/y_err))
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
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    https://www.tensorflow.org/guide/keras/custom_layers_and_models#putting_it_all_together_an_end-to-end_example"""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


original_dim = 128
intermediate_dim = 1024
latent_dim = 20

# Define encoder model.
original_inputs = tf.keras.Input(shape=(original_dim,1), name='encoder_input')
input_err = Input(shape=(original_dim,1))
x = layers.CuDNNLSTM(intermediate_dim, return_sequences=False)(original_inputs)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
z = Sampling()((z_mean, z_log_var))
encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name='encoder')

# Define decoder model.
latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.RepeatVector(original_dim)(latent_inputs)
x = layers.CuDNNLSTM(intermediate_dim, return_sequences=True)(x)
outputs = layers.TimeDistributed(layers.Dense(1))(x)
decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name='decoder')

# Define VAE model.
outputs = decoder(z)
vae = tf.keras.Model(inputs=[original_inputs, input_err], outputs=outputs, name='vae')

# Add KL divergence regularization loss.
kl_loss = - 0.5 * tf.reduce_mean(
    z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae.add_loss(kl_loss)

optimizer = tf.keras.optimizers.Adam(clipvalue=0.5)#learning_rate=1e-4, clipvalue=1e-4, clipnorm=1e-4)

vae.compile(optimizer, loss=chi2(input_err))


vae.metrics_tensors.append(kl_loss)
vae.metrics_names.append("kl_loss")


# Training and Validation generators
training_generator = DataGenerator(train_data_counts, train_data_errors, batch_size=1293)
validation_generator = DataGenerator(valid_data_counts, valid_data_errors, batch_size=1293) 

training_time_stamp = datetime.datetime.now(tz=pytz.timezone('Europe/London')).strftime("%Y-%m-%d_%H-%M-%S")

CB = EarlyStopping(monitor='val_loss', min_delta=5e-5, patience=50, verbose=1, mode='auto')
MC = ModelCheckpoint('../../models/VAE_weights/model_{}.h5'.format(training_time_stamp), monitor='val_loss', mode="auto", save_best_only=True, verbose=1)
history = vae.fit_generator(training_generator, epochs=8000, verbose=2, callbacks = [MC, CB], validation_data=validation_generator)


np.savetxt("../../reports/training_history/loss_history-{}.txt".format(training_time_stamp), [np.asarray(history.history["loss"]), np.asarray(history.history["val_loss"])], delimiter=",")
