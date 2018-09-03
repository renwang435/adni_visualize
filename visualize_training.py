import matplotlib.pyplot as plt
import pickle
import sys

if __name__ == '__main__':
    with open("cvae_training_dump.txt", "rb") as fp:  # Unpickling
        history = pickle.load(fp)

    # Visualize training history
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'val'], loc='best')
    plt.show()

    # Visualize training history
    plt.plot(history['kl_loss'])
    plt.plot(history['val_kl_loss'])
    plt.title('Model KL Loss')
    plt.ylabel('KL Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'val'], loc='best')
    plt.show()

    # Visualize training history
    plt.plot(history['recon_loss'])
    plt.plot(history['val_recon_loss'])
    plt.title('Model Reconstruction Loss')
    plt.ylabel('Reconstruction Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'val'], loc='best')
    plt.show()
