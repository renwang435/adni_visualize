# adni_visualize

A 3D convolutional neural network to classify fMRI images from the Alzheimer's Disease Neuroimaging Initiative (ADNI). Images fall into one of three categories:cognitively normal (CN), mild cognitive impairment (MCI) and Alzheimer's disease (AD).

Secondary goals: 
1. Investigate learned filter morphologies and overall manifolds at each layers
2. Explore Bayesian hyperparameter optimization

## ADNI_Preproc.py
Generates a set of .npy files after extracting the image data from a set of NifTi files

## ADNI_Train
Training script which 
	i) Specifies the CNN architecture
	ii) Initializes a set of data generators in order to use the fit_generator function in Keras
	iii) Train and evaluates the model

## DataGenerator.py
Class which specifies the data generation models expected for fit_generator
	