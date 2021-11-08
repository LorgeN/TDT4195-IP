from numpy.fft import fft
import skimage
import skimage.io
import skimage.transform
import os
import numpy as np
import utils
import matplotlib.pyplot as plt
from task4a import convolve_im, to_image


if __name__ == "__main__":
    # DO NOT CHANGE
    impath = os.path.join("images", "noisy_moon.png")
    im = utils.read_im(impath)

    # START YOUR CODE HERE ### (You can change anything inside this block)

    fft_kernel = np.ones_like(im)
    for i in range(26, im.shape[0] - 100, 29):
        fft_kernel[:4, i:(i + 6)] = 0
        fft_kernel[-3:, i:(i + 6)] = 0

    im_filtered = convolve_im(im, fft_kernel)
    plt.show()

    ### END YOUR CODE HERE ###
    utils.save_im("moon_filtered.png", utils.normalize(im_filtered))
