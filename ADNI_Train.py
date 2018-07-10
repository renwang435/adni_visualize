import glob
import pickle

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from DataGenerator import DataGenerator

write_dir = 'preprocessed_data'


def cnn_arch():
    model = Sequential()

    return model


if __name__ == '__main__':
    params = {'batch_size':1,
              'dim': (256, 256, 170),
              'n_channels': 1,
              'n_classes':3,
              'shuffle':True
              }

    # Read in dictionary mapping file numbers to labels
    with open('./file_num_to_tag.txt', 'rb') as fp:
        num_to_label = pickle.load(fp)

    # Generate range of .npy files and split into train, validation and test sets
    num_files = len(glob.glob(write_dir + '\\*.npy'))
    train_indices = np.arange(0, num_files)

    X_train, X_test, _, _ = train_test_split(train_indices,
                                             train_indices,
                                             test_size=0.2,
                                             random_state=42)

    X_val, X_test, _, _ = train_test_split(X_test,
                                           X_test,
                                           test_size=0.5,
                                           random_state=42)

    # Generators
    training_generator = DataGenerator(X_train, num_to_label, **params)
    validation_generator = DataGenerator(X_val, num_to_label, **params)
    evaluation_generator = DataGenerator(X_test, num_to_label, **params)

    # Callbacks
    checkpointer = ModelCheckpoint(filepath='bestmodel.hdf5',
                                   monitor='val_loss',
                                   save_best_only=True,
                                   verbose=1)

    early_stopper = EarlyStopping(monitor='val_loss',
                                  patience=10,
                                  verbose=1)