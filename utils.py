""" 
@Author: Kumar Nityan Suman
@Date: 2019-06-22 11:45:40
"""


# Load packages
import os
import gzip
import numpy as np

def load_mnist(path="data", kind="train"):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, "{}-labels-idx1-ubyte.gz".format(kind))
    images_path = os.path.join(path, "{}-images-idx3-ubyte.gz".format(kind))

    # Unzip data and read labels
    with gzip.open(labels_path, mode="rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    # Unzip data and read imaged as a numpy array
    with gzip.open(images_path, mode="rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels

def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    # Empty sprite image
    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))
    
    # Construct the sprite image
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img
    return spriteimage

def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits, (-1, 28, 28))

def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 255 - mnist_digits

def get_sprite_image(to_visualise, do_invert=True):
    to_visualise = vector_to_matrix_mnist(to_visualise)
    if do_invert:
        to_visualise = invert_grayscale(to_visualise)
    return create_sprite_image(to_visualise)