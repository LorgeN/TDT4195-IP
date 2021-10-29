import matplotlib.pyplot as plt
import pathlib
import numpy as np
from utils import read_im, save_im, normalize

output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


im = read_im(pathlib.Path("images", "lake.jpg"))
plt.imshow(im)


def convolve_im(
    im,
    kernel,
):
    """A function that convolves im with kernel

    Args:
        im ([type]): [np.array of shape [H, W, 3]]
        kernel ([type]): [np.array of shape [K, K]]

    Returns:
        [type]: [np.array of shape [H, W, 3]. should be same as im]
    """
    assert len(im.shape) == 3

    k = len(kernel)
    k2 = k // 2
    H, W, c = im.shape

    # Recreate an array padded with 0s so we dont have to worry about index out
    # of bounds at later stages
    src = np.zeros(np.array([H + k, W + k, c]))
    # Copy the image given as input into the center of our padded array
    src[k2 : (H + k2), k2 : (W + k2), :] = im

    # Create a new numpy array in the shape of the image to use as a result value
    res = np.zeros_like(im)

    for y in range(H):
        for x in range(W):
            # Extract the section of the source image that we need to multiply
            # with our kernel. Extracted flipped so we can do a convolution by
            # multiplying with our kernel. Alternatively we could flip the kernel.
            pix = src[(y + k) : y : -1, (x + k) : x : -1, :]

            # Iterate over the R, G and B components
            for i in range(c):
                res[y, x, i] = (pix[:, :, i] * kernel).sum()

    return res


if __name__ == "__main__":
    # Define the convolutional kernels
    h_b = (
        1
        / 256
        * np.array(
            [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1],
            ]
        )
    )
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Convolve images
    im_smoothed = convolve_im(im.copy(), h_b)
    save_im(output_dir.joinpath("im_smoothed.jpg"), im_smoothed)
    im_sobel = convolve_im(im, sobel_x)
    save_im(output_dir.joinpath("im_sobel.jpg"), im_sobel)

    # DO NOT CHANGE. Checking that your function returns as expected
    assert isinstance(im_smoothed, np.ndarray), (
        f"Your convolve function has to return a np.array. "
        + f"Was: {type(im_smoothed)}"
    )
    assert im_smoothed.shape == im.shape, (
        f"Expected smoothed im ({im_smoothed.shape}"
        + f"to have same shape as im ({im.shape})"
    )
    assert im_sobel.shape == im.shape, (
        f"Expected smoothed im ({im_sobel.shape}"
        + f"to have same shape as im ({im.shape})"
    )
    plt.subplot(1, 2, 1)
    plt.imshow(normalize(im_smoothed))

    plt.subplot(1, 2, 2)
    plt.imshow(normalize(im_sobel))
    plt.show()
