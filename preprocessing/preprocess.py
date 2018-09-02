from __future__ import division

import os
import pickle
import shutil
import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import zoom

# The purpose of this script is to take a directory of NifTI files and retrieve the relevant
# image data in a .npy format and save in an appropriate location
# We also pickle a dictionary mapping the image data files to their correct label

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return

def load_nii(path):
    return nib.load(path).get_data()

def resize(data, target_shape=[96, 112, 96]):
    factor = [float(t) / float(s) for t, s in zip(target_shape, data.shape)]
    resized = zoom(data, zoom=factor, order=1, prefilter=False)

    return resized

def preproc(path):
    nii_data = load_nii(path)

    return resize(nii_data)

def norm(data):
    data = data / float(np.max(data))

    return data

data_dir = '../data/ADNIDenoise'
write_dir = '../data/training_examples'
if not os.path.exists(write_dir):
    os.makedirs(write_dir)
else:
    shutil.rmtree(write_dir)
    create_dir(write_dir)

if __name__ == '__main__':
    data_labels = ['AD', 'MCI', 'CN']
    remap = dict()
    num_training_examples = 0
    for i, label in enumerate(data_labels):
        src_label_dir = os.path.join(data_dir, label)
        for subject in os.listdir(src_label_dir):
            curr_file = os.path.join(src_label_dir, subject)
            post_processed_nii = preproc(curr_file)
            post_processed_nii = norm(post_processed_nii)
            np.save(os.path.join(write_dir, str(num_training_examples)), post_processed_nii)
            remap[num_training_examples] = i
            num_training_examples += 1

    with open('../file_num_to_tag.txt', 'wb') as fp:
        pickle.dump(remap, fp, protocol=-1)
