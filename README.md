# adni_visualize

The aim to utlize image datasets from the Alzheimer's Disease Neuroimaging Initiative (ADNI) in various neural network learning tasks and investigate learned filter morphologies at each layer and overall manifolds. Images fall into one of three categories:cognitively normal (CN), mild cognitive impairment (MCI) and Alzheimer's disease (AD).

## architectures
Images depicting various architectures experimented with.
### cnn_arch.png
3D convolutional neural network
### vae_cnn_encoder.png
Encoder for convolutional variational autoencoder
### vae_cnn_decoder.png
Decoder for convolutional variational autoencoder
### vae_cnn.png
Overall architecture for convolutional variational autoencoder


## preprocessing
Set of scripts for preprocessing 
### compile_adni.py
Matches NiFTi files to the master metadata file provided by ADNI and segregates them to a folder denoting their appropriate class label (AD, MCI, CN)
### register.py
Performs affine registration to the MNI152-1mm-T1 atlas on images using FLIRT
### skull_strip.py
Performs skull stripping with FSL BET.
### bias_correct.py
Performs N4 enhanced bias correction using the N4ITK tool in ANTs.
### preprocess.py
Performs normalization on images and writes them into easily loadable NPY formats to Keras' fit_generator.

## adni_convnet.py
A 3D convolutional neural network for 1-of-3 classification of the images and relevant training code.

## adni_cvae.py
A 2D variational autoencoder and relevant training code (note that channels are set to the z-dimension of the images)

## CNNDataGenerator.py
Dynamic batch generator for adni_convnet.py. This is intended to be used as a data generator class for Keras' fit_generator.

## VAEDataGenerator.py
Dynamic batch generator for adni_cvae.py. This is intended to be used as a data generator class for Keras' fit_generator.

## latent_manifold.py
Visualizes the AD, MCI and CN t-SNE clusters when images are mapped to the latent space of the CVAE. t-SNE is initialized with PCA.

## visualize_training.py
Plots the relevant training losses over training epochs.

## visualize_brains.py
Small script to visualize a slice from the middle of the brain volume after it undergoes the preprocessing pipeline.
	