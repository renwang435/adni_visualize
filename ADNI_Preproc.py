from __future__ import division

import glob
import os
import sys

import pickle
import nibabel as nib
import numpy as np
import pandas as pd

# The purpose of this script is to take a directory of NifTI files and retrieve the relevant
# image data in a .npy format and save in an appropriate location
# We also pickle a dictionary mapping the image data files to their correct label

# We define a CN label as 0, MCI label as 1, and an AD label as 2, to be one-hot encoded in DataGenerator

master_df = pd.read_csv('ADNI1_Complete_2Yr_3T_7_09_2018.csv', skipinitialspace=True)
data_dir = 'adni_data'
write_dir = 'preprocessed_data'
if not os.path.exists(write_dir):
    os.makedirs(write_dir)

adni_files = glob.glob(data_dir + r'\*\MPR____N3__Scaled\*\*\*.nii')
i = 0
progress = 0
total = len(adni_files)
normalization = {}
remap = {}
for filename in adni_files:
    curr_nifti_data = np.array(nib.load(filename).get_data())

    image_data_id = filename.split('_')[-1].split('.')[0][1:]
    group = master_df[master_df['Image Data ID'] == int(image_data_id)]['Group'].values
    if (len(group) > 1):
        print('More than one Image Data ID corresponding to: ' + str(image_data_id))
        sys.exit(1)
    elif (len(group) < 0):
        print('Skipping...could not find an entry for this Image Data ID: ' + str(image_data_id))
        progress += 1
        continue
    group = group[0]

    if (group == 'CN'):
        label = 0
    elif (group == 'MCI'):
        label = 1
    elif (group == 'AD'):
        label = 2
    else:
        print('Unknown patient group: ' + group)
        sys.exit(1)

    # try:
    #     mean = np.mean(curr_nifti_data)
    #     std = np.std(curr_nifti_data)
    #     curr_nifti_data = (curr_nifti_data - mean) / std
    #     normalization[i] = (mean, std)
    # except ZeroDivisionError:
    #     print('Cannot standardize: ' + str(image_data_id))
    #     sys.exit(1)

    np.save(os.path.join(write_dir, str(i) + '.npy'), curr_nifti_data)
    remap[i] = label
    i += 1
    progress += 1

    print('Completed ' + str(progress / total * 100) + '%')

# with open('./file_num_to_normalization.txt', 'wb') as fp:
#     joblib.dump(normalization, fp, protocol=-1)

with open('./file_num_to_tag.txt', 'wb') as fp:
    pickle.dump(remap, fp, protocol=-1)

# 2. Look through Roger + general papers for fMRI convolutional architecture
# 3a. Write CNN architecture
# 3b. Email Roger
# 4a. Train and save model
# 4b. Write code to visualize filters
