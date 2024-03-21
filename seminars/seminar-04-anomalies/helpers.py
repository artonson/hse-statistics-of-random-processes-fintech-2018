import math
import sys
import os
import time

import numpy as np
import matplotlib.pyplot as plt


# Plots the image represented by a row
def plot_number(pixels, label=None, w=28, h=28):
    # Make those columns into a array of 8-bits pixels
    # This array will be of 1D with length 784
    # The pixel intensity values are integers from 0 to 255
    pixels = 255 - np.array(pixels, dtype='uint8')

    # Reshape the array into 28 x 28 array (2-dimensional array)
    pixels = pixels.reshape((w, h))

    # Plot
    if label is not None:
        plt.title('Label is {label}'.format(label=label))
    plt.imshow(pixels, cmap='gray')


# Plots a whole slice of pictures
def plot_slice(rows, labels, size_w=28, size_h=28):
    num = rows.shape[0]
    w = 4
    h = int(math.ceil(num / w))
    fig, plots = plt.subplots(h, w)
    fig.tight_layout()

    for n in range(0, num):
        s = plt.subplot(h, w, n+1)
        s.set_xticks(())
        s.set_yticks(())
        plot_number(rows[n], labels[n], size_w, size_h)
    plt.show()

