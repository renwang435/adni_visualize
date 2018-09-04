from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import multiprocessing as mp
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.layers import Conv2D, MaxPool2D, UpSampling2D, Flatten, Lambda
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.layers import Reshape
from keras.losses import mse
from keras.models import Model
from keras.optimizers import Adam
from sklearn.manifold import TSNE

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

def retrieve_classes(X, loaded_examples, num_to_label):
    list_of_classes = []
    for example in loaded_examples:
        file_num = str(example).split('\\')[-1].split('.')[0]
        list_of_classes.append(num_to_label[int(file_num)])

    return np.array(list_of_classes)


# Show dataset images with T-sne projection of latent space encoding
def computeTSNEProjectionOfLatentSpace(X, loaded_examples, model, num_to_label,
                                       save=True, display=True):
    if save:
        # Compute latent space representation
        print("Computing latent space projections...")
        # Encoder returns 3 values, VAE returns 1 value
        try:
            _, _, X_encoded = model.predict(X)
        except:
            print('Assuming one output only from model')
            X_encoded = model.predict(X)

        # PCA and TSNE expect 2D input
        X_encoded = X_encoded.reshape((np.shape(X_encoded)[0], -1))

        # Compute t-SNE embedding of latent space
        # Intermediate dimensionality reduction with PCA first
        print('Computing TSNE...')
        tsne = TSNE(n_components=2, verbose=1, random_state=0, init='pca',
                    perplexity=40, n_iter=1000)
        X_tsne = tsne.fit_transform(X_encoded)
        np.save('X_tsne', X_tsne)
    else:
        print('Loading previously computed tsne vectors...')
        X_tsne = np.load('X_tsne.npy')

    color_map = retrieve_classes(X, loaded_examples, num_to_label)
    colors = []
    for i in color_map:
        colors.append(i)
    colors = np.array(colors)

    scatter_x = X_tsne[:, 0]
    scatter_y = X_tsne[:, 1]
    cdict = {0: 'red', 1: 'yellow', 2: 'green'}
    labeldict = {0: 'AD', 1: 'MCI', 2: 'CN'}

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots()
        for c in np.unique(colors):
            ix = np.where(colors == c)
            ax.scatter(scatter_x[ix], scatter_y[ix], c=cdict[c], label=labeldict[c])
        ax.legend()
        plt.show()
    else:
        print("Saving t-SNE visualization in PNG format...")
        fig, ax = plt.subplots()
        for c in np.unique(colors):
            ix = np.where(colors == c)
            ax.scatter(scatter_x[ix], scatter_y[ix], c=cdict[c], label=labeldict[c])
        ax.legend()
        plt.savefig('tsne_no_encoding.png')

def load_examples(file, inputs_for_tsne):
    sample = np.array(np.load(file))
    input = np.transpose(sample)
    inputs_for_tsne[file] = input

def load_examples_star(all_args):
    return load_examples(*all_args)

if __name__ == '__main__':
    params = {'batch_size': 32,
              'dim': (96, 112),
              'n_channels': 96,
              'n_classes': 3,
              'shuffle': True
              }
    latent_dim = 15
    epochs = 200
    num_training = 815
    use_mp = True
    workers = mp.cpu_count()
    with open('./file_num_to_tag.txt', 'rb') as fp:
        num_to_label = pickle.load(fp)

    # VAE model = encoder + decoder
    # build encoder model
    inputs, z_mean, z_log_var, z, shape = encoder_arch((*params['dim'], params['n_channels']))
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # build decoder model
    latent_inputs, outputs = decoder_arch(latent_dim)
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])

    # Testing
    vae = Model(inputs, outputs, name='vae')
    # Load best weights
    vae.load_weights('cvae_bestmodel.h5')
    # Recompile model
    print('compiling model')
    custom_opt = Adam(lr=0.000027, clipvalue=0.5)
    vae.compile(optimizer=custom_opt, loss=vae_loss, metrics=[recon_loss,
                                                                  kl_loss])

    data_dir = 'data/training_examples'
    segments = glob.glob(data_dir + '/*.npy')[:num_training]
    inputs_for_tsne = mp.Manager().dict()
    print('starting tsne transformation on ' + str(len(segments)) + ' encodings')

    # num_threads = mp.cpu_count()
    # pool = mp.Pool(processes=num_threads)
    # pool.map_async(load_examples_star,
    #                zip(segments,
    #                    itertools.repeat(inputs_for_tsne)))
    # pool.close()
    # pool.join()
    #
    # dict_segments = dict(inputs_for_tsne)
    #
    # loaded_values = list(dict_segments.values())
    # loaded_examples = list(dict_segments.keys())
    # loaded_values = np.array(loaded_values)
    # loaded_examples = np.array(loaded_examples)
    #
    # np.save('loaded_values_subset.npy', loaded_values)
    # np.save('loaded_examples_subset.npy', loaded_examples)
    loaded_values = np.load('loaded_values_subset.npy')
    loaded_examples = np.load('loaded_examples_subset.npy')
    print('Done loading ' + str(len(loaded_values)) + ' examples')

    computeTSNEProjectionOfLatentSpace(loaded_values, loaded_examples, encoder, num_to_label,
                                       save=False, display=False)