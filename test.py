# Script for playing around with experimental ideas

import nibabel as nib
import numpy as np
import sys
import pickle

if __name__ == '__main__':
    with open('file_num_to_tag.txt', 'rb') as fp:
        num_to_tag = pickle.load(fp)

    vals = np.array(list(num_to_tag.values()))
    # print(vals)
    print(len(np.where(vals == 0)[0]))
    print(len(np.where(vals == 1)[0]))
    print(len(np.where(vals == 2)[0]))

    # print(num_to_tag)