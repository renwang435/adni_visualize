# Script for playing around with experimental ideas

import nibabel as nib
import numpy as np
import sys

if __name__ == '__main__':
    nifti_data = nib.load('ADNI_136_S_0429_MR_MPR____N3__Scaled_Br_20071127184003417_S33730_I83549.nii')
    nifti_data = np.array(nifti_data.get_data())
    # print(nifti_data)
    # sys.exit(1)
    #
    count = 0
    for i in range(len(nifti_data)):
        for j in range(len(nifti_data[i])):
            for k in range(len(nifti_data[i][j])):
                # if (nifti_data[i][j][k] > 0 and nifti_data[i][j][k] <= 1):
                if (nifti_data[i][j][k] > 1):
                    count += 1

    print(count)
