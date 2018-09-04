import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nibabel as nib
import numpy as np

if __name__ == '__main__':
    test = np.load('data/training_examples/3.npy')
    mid_slice = test[:, :, test.shape[-1] // 2]
    plt.imshow(mid_slice)
    plt.show()