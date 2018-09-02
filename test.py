# Script for playing around with experimental ideas

import nibabel as nib
import numpy as np
import os
import sys
import pickle

parent_dir = 'temp'
labels = ['AD', 'CN', 'MCI']

if __name__ == '__main__':
    for i in range(815):
        test = np.load('data/training_examples/' + str(i) + '.npy')
        if (test.shape != (96, 112, 96)):
            print(i)

    # partition = np.arange(0, 815)
    # print(partition)