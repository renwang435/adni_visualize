import DataGenerator
import pickle
from keras.models import Model, Sequential


def cnn_arch():
    model = Sequential()

    return model


if __name__ == '__main__':
    params = {'batch_size':1,
              'dim': (32, 32, 32),
              'n_channels': 1,
              'n_classes':3,
              'shuffle':True
              }

    # Read in dictionary mapping file numbers to labels
    with open('./file_num_to_tag.txt', 'rb') as fp:
        num_to_label = pickle.load(fp)

    # Generate range of .npy files and split into train, validation and test sets

