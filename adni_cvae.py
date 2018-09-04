from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing as mp
import os
import sys
import pickle

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, MaxPool2D, UpSampling2D, Flatten, Lambda
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.layers import Reshape
from keras.losses import mse
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split

from VAEDataGenerator import DataGenerator

def swish(affine):
    return affine * K.sigmoid(affine)

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))

    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def recon_loss(y_true, y_pred):
    return mse(K.flatten(y_true), K.flatten(y_pred))

def kl_loss(y_true, y_pred):
    return -0.5 * K.sum(1. + z_log_var - K.exp(z_log_var) - K.square(z_mean), axis=-1)

def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = recon_loss(y_true, y_pred)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = kl_loss(y_true, y_pred)

    return recon + 2 * kl

def encoder_arch(input_shape):
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs

    x = Conv2D(filters=96,
               kernel_size=(12, 14),
               activation=swish,
               strides=(2, 2),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=192,
               kernel_size=(6, 7),
               activation=swish,
               strides=(1, 1),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=192,
               kernel_size=(3, 3),
               activation=swish,
               strides=(1, 1),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=384,
               kernel_size=(3, 3),
               activation=swish,
               strides=(1, 1),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=384,
               kernel_size=(3, 3),
               activation=swish,
               strides=(1, 1),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=768,
               kernel_size=(3, 3),
               activation=swish,
               strides=(1, 1),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)

    shape = np.array(K.int_shape(x))

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(1024, activation=swish)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation=swish)(x)

    z_mean = Dense(latent_dim, activation='linear', name='z_mean')(x)
    z_log_var = Dense(latent_dim, activation='linear', name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    return inputs, z_mean, z_log_var, z, shape

def decoder_arch(latent_dim):
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(128, activation=swish)(latent_inputs)
    x = BatchNormalization()(x)
    x = Dense(1024, activation=swish)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(shape[1] * shape[2] * shape[3], activation=swish)(x)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    x = Conv2D(filters=768,
               kernel_size=(3, 3),
               activation=swish,
               strides=(1, 1),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=384,
               kernel_size=(3, 3),
               activation=swish,
               strides=(1, 1),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D()(x)

    x = Conv2D(filters=384,
               kernel_size=(3, 3),
               activation=swish,
               strides=(1, 1),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=192,
               kernel_size=(3, 3),
               activation=swish,
               strides=(1, 1),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D()(x)

    x = Conv2D(filters=192,
               kernel_size=(6, 7),
               activation=swish,
               strides=(1, 1),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D()(x)


    x = Conv2D(filters=96,
                     kernel_size=(12, 14),
                     activation='sigmoid',
                     strides=(1, 1),
                     padding='same')(x)
    x = BatchNormalization()(x)
    outputs = UpSampling2D(name='decoder_output')(x)

    return latent_inputs, outputs

if __name__ == '__main__':
    params = {'batch_size': 32,
              'dim': (96, 112),
              'n_channels': 96,
              'n_classes': 3,
              'shuffle': True
              }
    latent_dim = 15
    epochs = 200
    use_mp = True
    workers = mp.cpu_count()
    with open('./file_num_to_tag.txt', 'rb') as fp:
        num_to_label = pickle.load(fp)

    num_training = 815

    # VAE model = encoder + decoder
    # build encoder model
    inputs, z_mean, z_log_var, z, shape = encoder_arch((*params['dim'], params['n_channels']))
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    print(encoder.summary())
    plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs, outputs = decoder_arch(latent_dim)
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    print(decoder.summary())
    plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    # Compile model
    custom_opt = Adam(lr=0.000027, clipvalue=0.5)
    vae.compile(optimizer=custom_opt, loss=vae_loss, metrics=[recon_loss,
                                                                  kl_loss])
    print(vae.summary())
    plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

    # Datasets
    partition = np.arange(0, num_training)
    X_train, X_test, _, _ = train_test_split(partition,
                                             partition,
                                             test_size=0.2,
                                             random_state=42)

    X_val, X_test, _, _ = train_test_split(X_test,
                                           X_test,
                                           test_size=0.5,
                                           random_state=42)

    # Generators
    training_generator = DataGenerator(X_train, **params)
    validation_generator = DataGenerator(X_val, **params)
    evaluation_generator = DataGenerator(X_test, **params)

    # Callbacks
    checkpointer = ModelCheckpoint(filepath='cvae_bestmodel.h5',
                                   monitor='val_loss',
                                   save_best_only=True,
                                   verbose=1)
    early_stopper = EarlyStopping(monitor='val_loss',
                                  patience=10,
                                  verbose=1)

    # Train model on dataset
    autoencoder_train_history = vae.fit_generator(generator=training_generator,
                                                  validation_data=validation_generator,
                                                  epochs=epochs,
                                                  use_multiprocessing=use_mp,
                                                  verbose=1,
                                                  callbacks=[checkpointer, early_stopper],
                                                  workers=1)

    # Testing
    vae = Model(inputs, outputs, name='vae')
    # Load best weights
    vae.load_weights('cvae_bestmodel.h5')
    # Recompile model
    print('compiling model')
    vae.compile(optimizer=custom_opt, loss=vae_loss, metrics=[recon_loss,
                                                                  kl_loss])

    tresults = vae.evaluate_generator(generator=evaluation_generator,
                                      workers=1,
                                      use_multiprocessing=use_mp,
                                      verbose=1)
    print('Overall loss: ' + str(tresults[0]))

    with open("./cvae_training_dump.txt", "wb") as fp:  # Pickling
        pickle.dump(autoencoder_train_history.history, fp, protocol=2)