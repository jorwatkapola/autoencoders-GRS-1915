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
# from nima import emd_loss

from tensorflow.keras.backend import flatten
from tensorflow.keras.backend import concatenate
from tensorflow.keras.backend import reverse
from tensorflow import cumsum





np.random.seed(seed=11)

with open('../../data_GRS1915/468202_len128_s2_4cad_histograms_24bin_0-13k_errorfix.pkl', 'rb') as f:
    segments = pickle.load(f)

segments = zscore(segments, axis=None).astype(np.float32)  # standardize


def tril_indices(n, k=0):
    """Return the indices for the lower-triangle of an (n, m) array.
    Works similarly to `np.tril_indices`
    Args:
      n: the row dimension of the arrays for which the returned indices will
        be valid.
      k: optional diagonal offset (see `np.tril` for details).
    Returns:
      inds: The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.
    """
    m1 = tf.tile(tf.expand_dims(tf.range(n), axis=0), [n, 1])
    m2 = tf.tile(tf.expand_dims(tf.range(n), axis=1), [1, n])
    mask = (m1 - m2) >= -k
    ix1 = tf.boolean_mask(m2, tf.transpose(mask))
    ix2 = tf.boolean_mask(m1, tf.transpose(mask))
    return ix1, ix2

def ecdf(p):
    """Estimate the cumulative distribution function.
    The e.c.d.f. (empirical cumulative distribution function) F_n is a step
    function with jump 1/n at each observation (possibly with multiple jumps
    at one place if there are ties).
    For observations x= (x_1, x_2, ... x_n), F_n is the fraction of
    observations less or equal to t, i.e.,
    F_n(t) = #{x_i <= t} / n = 1/n \sum^{N}_{i=1} Indicator(x_i <= t).
    Args:
      p: a 2-D `Tensor` of observations of shape [batch_size, num_classes].
        Classes are assumed to be ordered.
    Returns:
      A 2-D `Tensor` of estimated ECDFs.
    """
    n = p.get_shape().as_list()[1]
    indices = tril_indices(n)
    indices = tf.transpose(tf.stack([indices[1], indices[0]]))
    ones = tf.ones([n * (n + 1) / 2])
    triang = tf.scatter_nd(indices, ones, [n, n])
    return tf.matmul(p, triang)

def emd_loss(p, p_hat, r=2, scope=None):
    """Compute the Earth Mover's Distance loss.
    Hou, Le, Chen-Ping Yu, and Dimitris Samaras. "Squared Earth Mover's
    Distance-based Loss for Training Deep Neural Networks." arXiv preprint
    arXiv:1611.05916 (2016).
    Args:
      p: a 2-D `Tensor` of the ground truth probability mass functions.
      p_hat: a 2-D `Tensor` of the estimated p.m.f.-s
      r: a constant for the r-norm.
      scope: optional name scope.
    `p` and `p_hat` are assumed to have equal mass as \sum^{N}_{i=1} p_i =
    \sum^{N}_{i=1} p_hat_i
    Returns:
      A 0-D `Tensor` of r-normed EMD loss.
    """
    with tf.name_scope(scope, 'EmdLoss', [p, p_hat]):
        ecdf_p = ecdf(p)
        ecdf_p_hat = ecdf(p_hat)
        emd = tf.reduce_mean(tf.pow(tf.abs(ecdf_p - ecdf_p_hat), r), axis=-1)
        emd = tf.pow(emd, 1 / r)
        return tf.reduce_mean(emd)
    



# def chi2(y_err):
#     def MSE_scaled(y_in, y_out,):
#         return mean(square(y_in-y_out)/square(y_err))
#     return MSE_scaled


# class DataGenerator(Sequence):
#     """
#     Generates data for Keras
#     https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
#     https://stackoverflow.com/questions/53105294/implementing-a-batch-dependent-loss-in-keras
#     """
#     def __init__(self, y_in, y_err, batch_size=32, shuffle=True):
#         'Initialization'
#         self.batch_size = batch_size
#         self.y_in = y_in
#         self.y_err = y_err        
#         self.shuffle = shuffle
#         self.on_epoch_end()

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.y_in) / self.batch_size))

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#         # Find list of IDs
#         y_in = self.y_in[indexes]
#         y_err = self.y_err[indexes]
#         return [y_in, y_err], y_in
    
#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.y_in))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    https://www.tensorflow.org/guide/keras/custom_layers_and_models#putting_it_all_together_an_end-to-end_example"""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


original_dim = 24
intermediate_dim = 64
latent_dim = 16



index_tensors = np.tile(np.arange(original_dim), (len(segments),1))






# Define encoder model.
original_inputs = tf.keras.Input(shape=(original_dim,1), name='encoder_input')


#
input_index = Input(shape=(original_dim,1))

#


# input_err = Input(shape=(original_dim,1))
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
vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name='vae')

# Add KL divergence regularization loss.
kl_loss = - 0.5 * tf.reduce_mean(
    z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae.add_loss(kl_loss)


########
# input_len = original_inputs.get_shape().as_list()[0]
# p_hat = concatenate((outputs, tf.keras.Input(tf.expand_dims(tf.range(original_dim, dtype="float32"),axis=1))), axis=-1)


p = concatenate((original_inputs, input_index), axis=-1)
p_hat = concatenate((outputs, input_index), axis=-1)


EMD = emd_loss(p, p_hat)

vae.add_loss(EMD)
########

optimizer = tf.keras.optimizers.Adam(clipvalue=0.5) #Adam(clipvalue=0.5)

vae.compile(optimizer, loss="mean_squared_error")

# vae.load_weights("../../model_weights/model_2020-04-24_13-14-37.h5")


    
# Training and Validation generators in a 95/5 split
# training_generator = DataGenerator(segments[:int(np.floor(len(segments)*0.95))], errors[:int(np.floor(len(errors)*0.95))], batch_size=2048)
# validation_generator = DataGenerator(segments[int(np.floor(len(segments)*0.95)):], errors[int(np.floor(len(errors)*0.95)):], batch_size=2048)    


training_time_stamp = datetime.datetime.now(tz=pytz.timezone('Europe/London')).strftime("%Y-%m-%d_%H-%M-%S")


CB = EarlyStopping(monitor='val_loss', min_delta=5e-5, patience=1000, verbose=1, mode='auto')
MC = ModelCheckpoint('../../model_weights/model_{}.h5'.format(training_time_stamp), monitor='val_loss', mode="auto", save_best_only=True, verbose=1)
history = vae.fit(x=[segments, index_tensors], y=segments, batch_size=4096, epochs=8000, verbose=2, callbacks = [MC, CB], validation_split=0.05)


np.savetxt("training_history/loss_history-{}.txt".format(training_time_stamp), [np.asarray(history.history["loss"]), np.asarray(history.history["val_loss"])], delimiter=",")
