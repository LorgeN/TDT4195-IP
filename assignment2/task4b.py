import matplotlib.pyplot as plt
import numpy as np
import skimage
import utils

idx = 1


def magnitude(fft_im):
    real = fft_im.real
    imag = fft_im.imag
    return np.sqrt(real ** 2 + imag ** 2)


def to_image(obj):
    return np.log(magnitude(obj) + 1)


def convolve_im(im: np.array,
                kernel: np.array,
                verbose=True):
    global idx
    """ Convolves the image (im) with the spatial kernel (kernel),
        and returns the resulting image.

        "verbose" can be used for turning on/off visualization
        convolution

        Note: kernel can be of different shape than im.

    Args:
        im: np.array of shape [H, W]
        kernel: np.array of shape [K, K] 
        verbose: bool
    Returns:
        im: np.array of shape [H, W]
    """
    kernel_pad = np.pad(kernel, [(0, im.shape[i] - kernel.shape[i]) for i in range(2)])

    fft_kernel = np.fft.fft2(kernel_pad)

    fft_im = np.fft.fft2(im)
    filtered = fft_im * fft_kernel

    conv_result = np.fft.ifft2(filtered).real
    if verbose:
        vis_fft_im = np.fft.fftshift(fft_im)
        vis_fft_kernel = np.fft.fftshift(fft_kernel)
        vis_filtered = np.fft.fftshift(filtered)

        # Use plt.subplot to place two or more images beside eachother
        plt.figure(figsize=(20, 4))
        # plt.subplot(num_rows, num_cols, position (1-indexed))
        plt.subplot(1, 5, 1)
        plt.imshow(im, cmap="gray")
        plt.subplot(1, 5, 2)
        # Visualize FFT
        plt.imshow(to_image(vis_fft_im), cmap="gray")
        plt.subplot(1, 5, 3)
        # Visualize FFT kernel
        plt.imshow(to_image(vis_fft_kernel), cmap="gray")
        plt.subplot(1, 5, 4)
        # Visualize filtered FFT image
        plt.imshow(to_image(vis_filtered), cmap="gray")
        plt.subplot(1, 5, 5)
        # Visualize filtered spatial image
        plt.imshow(conv_result, cmap="gray")
        plt.savefig(utils.image_output_dir.joinpath(f"task4b_{idx}.png"))
        idx += 1

    return conv_result


if __name__ == "__main__":
    verbose = True  # change if you want

    # Changing this code should not be needed
    im = skimage.data.camera()
    im = utils.uint8_to_float(im)

    # DO NOT CHANGE
    gaussian_kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]) / 256
    image_gaussian = convolve_im(im, gaussian_kernel, verbose)

    # DO NOT CHANGE
    sobel_horizontal = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    image_sobelx = convolve_im(im, sobel_horizontal, verbose)

    if verbose:
        plt.show()

    utils.save_im("camera_gaussian.png", image_gaussian)
    utils.save_im("camera_sobelx.png", image_sobelx)
