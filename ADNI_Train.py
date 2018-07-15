import glob
import multiprocessing as mp
import pickle

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv3D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

from DataGenerator import DataGenerator

write_dir = 'preprocessed_data'


#TODO: Architecture search for optimal convolutional structure
def cnn_arch():
    model = Sequential()
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), input_shape=(256, 256, 170, 1)))

    # Hyperparameters for additional convolutional layers

    model.add(Flatten())

    # 3-way softmax for predicting AD, MCI and CN classes
    model.add(Dense(3, activation='softmax'))

    return model


if __name__ == '__main__':
    params = {'batch_size':1,
              'dim': (256, 256, 170),
              'n_channels': 1,
              'n_classes':3,
              'shuffle':True
              }
    epochs = 100
    use_mp = True
    workers = mp.cpu_count()

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

    model = cnn_arch()
    # Compile model
    print('Compiling model')
    custom_rmsprop = RMSprop(lr=0.001, clipvalue=0.5)
    model.compile(optimizer=custom_rmsprop,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    train_history = model.fit_generator(generator=training_generator,
                                        validation_data=validation_generator,
                                        epochs=epochs,
                                        use_multiprocessing=use_mp,
                                        verbose=1,
                                        callbacks=[checkpointer, early_stopper],
                                        workers=workers)
    
    # Testing
    model = cnn_arch()
    # Load best weights
    model.load_weights('bestmodel.hdf5')
    # Recompile model
    model.compile(optimizer=custom_rmsprop,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    tresults = model.evaluate_generator(generator=evaluation_generator,
                                        workers=workers,
                                        use_multiprocessing=use_mp,
                                        verbose=1)

    print('Overall accuracy: ' + str(tresults[1]))

    with open('./training_dump.txt', 'wb') as fp:
        pickle.dump(train_history.history, fp, protocol=-1)