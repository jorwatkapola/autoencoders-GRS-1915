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

with open('../../data_GRS1915/468202_len128_s2_4cad_counts_errorfix.pkl', 'rb') as f:
    segments = pickle.load(f)
with open('../../data_GRS1915/468202_len128_s2_4cad_errors_errorfix.pkl', 'rb') as f:
    errors = pickle.load(f)

# errors = np.expand_dims((np.squeeze(errors)/(np.max(segments, axis=1)-np.min(segments, axis=1))), axis=-1).astype(np.float32)
# segments = np.expand_dims(((np.squeeze(segments)-np.min(segments, axis=1))/(np.max(segments, axis=1)-np.min(segments, axis=1))), axis=-1).astype(np.float32)
# errors = ((errors)/np.std(segments)).astype(np.float32)
# segments = zscore(segments, axis=None).astype(np.float32)  # standardize


errors = ((errors)/np.expand_dims(np.std(segments, axis=1), axis=1)).astype(np.float32)
segments = zscore(segments, axis=1).astype(np.float32)  # standardize per segment


with open('../../data_GRS1915/468202_len128_s2_4cad_observation90-10split.pkl', 'rb') as f:
    split_segment_indices = pickle.load(f)


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
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    https://www.tensorflow.org/guide/keras/custom_layers_and_models#putting_it_all_together_an_end-to-end_example"""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def tc_penalty(epoch_no, z_sampled, z_mean, z_log_squared_scale, prior="normal"):
    """
    Source: https://github.com/julian-carpenter/beta-TCVAE/blob/master/nn/losses.py
    https://github.com/AntixK/PyTorch-VAE/blob/master/models/betatc_vae.py
    From:
    Locatello, F. et al.
    Challenging Common Assumptions in the Unsupervised Learning
    of Disentangled Representations. (2018).
    Based on Equation 4 with alpha = gamma = 1 of "Isolating Sources of
    Disentanglement in Variational Autoencoders"
    (https://arxiv.org/pdf/1802.04942).
    If alpha = gamma = 1, Eq. 4 can be written as ELBO + (1 - beta) * TC.
    --
    :param args: Shared arguments
    :param z_sampled: Samples from latent space
    :param z_mean: Means of z
    :param z_log_squared_scale: Logvars of z
    :return: Total correlation penalty
    """
    anneal_steps=50
    tc = total_correlation(z_sampled, z_mean, z_log_squared_scale, prior)
    anneal_rate = min(0 + 1 * epoch_no / anneal_steps, 1)

    return anneal_rate * tc

def gaussian_log_density(samples, mean, log_squared_scale):
    """Source: https://github.com/julian-carpenter/beta-TCVAE/blob/master/nn/losses.py"""
    pi = tf.constant(np.pi)
    normalization = tf.log(2. * pi)
    inv_sigma = tf.exp(-log_squared_scale)
    tmp = (samples - mean)
    return -0.5 * (tmp * tmp * inv_sigma + log_squared_scale + normalization)


def total_correlation(z, z_mean, z_log_squared_scale, prior):
    """Source: https://github.com/julian-carpenter/beta-TCVAE/blob/master/nn/losses.py
    Estimate of total correlation on a batch.
    We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
    log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
    for the minimization. The constant should be equal to (num_latents - 1) *
    log(batch_size * dataset_size)
    Args:
      z: [batch_size, num_latents]-tensor with sampled representation.
      z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
      z_log_squared_scale: [batch_size, num_latents]-tensor with log variance of the encoder.
    Returns:
      Total correlation estimated on a batch.
    """
    # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
    # tensor of size [batch_size, batch_size, num_latents]. In the following
    # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
    if prior.lower() == "normal":
        log_qz_prob = gaussian_log_density(
            tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
            tf.expand_dims(z_log_squared_scale, 0))
    if prior.lower() == "laplace":
        log_qz_prob = laplace_log_density(
            tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
            tf.expand_dims(z_log_squared_scale, 0))
    # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
    # + constant) for each sample in the batch, which is a vector of size
    # [batch_size,].
    log_qz_product = tf.reduce_sum(
        tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
        axis=1,
        keepdims=False)
    # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
    # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
    log_qz = tf.reduce_logsumexp(
        tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
        axis=1,
        keepdims=False)
    return tf.reduce_mean(log_qz - log_qz_product)


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        anneal_steps=50
        def tc_penalty(epoch, z_sampled, z_mean, z_log_squared_scale, prior="normal"):
            """
            Source: https://github.com/julian-carpenter/beta-TCVAE/blob/master/nn/losses.py
            https://github.com/AntixK/PyTorch-VAE/blob/master/models/betatc_vae.py
            From:
            Locatello, F. et al.
            Challenging Common Assumptions in the Unsupervised Learning
            of Disentangled Representations. (2018).
            Based on Equation 4 with alpha = gamma = 1 of "Isolating Sources of
            Disentanglement in Variational Autoencoders"
            (https://arxiv.org/pdf/1802.04942).
            If alpha = gamma = 1, Eq. 4 can be written as ELBO + (1 - beta) * TC.
            --
            :param args: Shared arguments
            :param z_sampled: Samples from latent space
            :param z_mean: Means of z
            :param z_log_squared_scale: Logvars of z
            :return: Total correlation penalty
            """
            
            tc = total_correlation(z_sampled, z_mean, z_log_squared_scale, prior)
            anneal_rate = min(0 + 1 * epoch / anneal_steps, 1)

            return anneal_rate * tc
        print("anneal_rate altered to {}".format(min(0 + 1 * epoch / anneal_steps, 1)))
        

#             if self.include_mutinfo:
#                 modified_elbo = logpx - \
#                     (logqz_condx - logqz) - \
#                     self.beta * (logqz - logqz_prodmarginals) - \
#                     (1 - self.lamb) * (logqz_prodmarginals - logpz)
#             else:
#                 modified_elbo = logpx - \
#                     self.beta * (logqz - logqz_prodmarginals) - \
#                     (1 - self.lamb) * (logqz_prodmarginals - logpz)
    
# def rate_scheduler(global_step, num_warmup_steps, init_lr, learning_rate):
#     """https://github.com/julian-carpenter/beta-TCVAE/blob/master/nn/util.py"""
#     global_steps_int = tf.dtypes.cast(global_step, tf.int32)
#     warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

#     global_steps_float = tf.dtypes.cast(global_steps_int, tf.float32)
#     warmup_steps_float = tf.dtypes.cast(warmup_steps_int, tf.float32)

#     warmup_percent_done = global_steps_float / warmup_steps_float
#     warmup_learning_rate = init_lr * warmup_percent_done

#     is_warmup = tf.dtypes.cast(global_steps_int < warmup_steps_int, tf.float32)
#     return (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate
    
    
# args.annealed_beta = rate_scheduler(args.global_step,
#                                     int(args.steps_per_epoch * args.epochs / 2),
#                                     args.beta,
#                                     args.beta) + 1.




original_dim = 128
intermediate_dim = 1024
latent_dim = 16

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

# total correlation
epoch_no=1
tc_loss = -20* tc_penalty(epoch_no, Sampling()((z_mean, z_log_var)), z_mean, z_log_var, prior="normal")
vae.add_loss(tc_loss)

# elbo = tf.math.add(ae_loss, kl_loss, name="elbo")
# loss = tf.math.add(elbo, tc_loss, name="loss")
    
    
optimizer = tf.keras.optimizers.Adam(clipvalue=0.5) # SGD(lr=3e-4, clipvalue=0.5)

vae.compile(optimizer, loss=chi2(input_err))


vae.metrics_tensors.append(kl_loss)
vae.metrics_names.append("kl_loss")

vae.metrics_tensors.append(tc_loss)
vae.metrics_names.append("tc_loss")

# vae.metrics_tensors.append(chi2_nonfunc)
# vae.metrics_names.append("chi2_loss")

# vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')
# vae.add_metric(chi2(input_err), name='mse_loss', aggregation='mean')



# vae.load_weights("../../model_weights/model_2020-08-26_14-16-02.h5")


    
# Training and Validation generators in a 90/10 split
training_generator = DataGenerator(segments[split_segment_indices[0]], errors[split_segment_indices[0]], batch_size=1024)
validation_generator = DataGenerator(segments[split_segment_indices[1]], errors[split_segment_indices[1]], batch_size=1024)    


training_time_stamp = datetime.datetime.now(tz=pytz.timezone('Europe/London')).strftime("%Y-%m-%d_%H-%M-%S")

CB = EarlyStopping(monitor='val_loss', min_delta=5e-5, patience=50, verbose=1, mode='auto')
MC = ModelCheckpoint('../../model_weights/model_{}.h5'.format(training_time_stamp), monitor='val_loss', mode="auto", save_best_only=True, verbose=1)
history = vae.fit_generator(training_generator, epochs=8000, verbose=2, callbacks = [MC, CB, CustomCallback()], validation_data=validation_generator)


np.savetxt("training_history/loss_history-{}.txt".format(training_time_stamp), [np.asarray(history.history["loss"]), np.asarray(history.history["val_loss"])], delimiter=",")
