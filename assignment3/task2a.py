import numpy as np
import skimage
import utils
import pathlib
import functools


def otsu_thresholding(im: np.ndarray) -> int:
    """
    Otsu's thresholding algorithm that segments an image into 1 or 0 (True or False)
    The function takes in a grayscale image and outputs a boolean image

    args:
        im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
    return:
        (int) the computed thresholding value
    """
    assert im.dtype == np.uint8
    # START YOUR CODE HERE ### (You can change anything inside this block)
    # You can also define other helper functions
    # Compute normalized histogram
    hist, intensities = skimage.exposure.histogram(im, normalize=True)

    P_1 = np.array([sum(hist[i] for i in range(k)) for k in range(1, len(hist) + 1)])
    m_k = np.array(
        [sum(i * hist[i] for i in range(k)) for k in range(1, len(hist) + 1)]
    )

    m_G = m_k[-1]

    sigma2 = ((m_G * P_1 - m_k) ** 2) / (P_1 * (1 - P_1))
    keys = np.where(sigma2 == sigma2.max())
    return sum(intensities[key] for key in keys) / len(keys)


if __name__ == "__main__":
    # DO NOT CHANGE
    impaths_to_segment = [
        pathlib.Path("thumbprint.png"),
        pathlib.Path("polymercell.png"),
    ]
    for impath in impaths_to_segment:
        im = utils.read_image(impath)
        threshold = otsu_thresholding(im)
        print("Found optimal threshold:", threshold)

        # Segment the image by threshold
        segmented_image = im >= threshold
        assert (
            im.shape == segmented_image.shape
        ), "Expected image shape ({}) to be same as thresholded image shape ({})".format(
            im.shape, segmented_image.shape
        )
        assert (
            segmented_image.dtype == np.bool
        ), "Expected thresholded image dtype to be np.bool. Was: {}".format(
            segmented_image.dtype
        )

        segmented_image = utils.to_uint8(segmented_image)

        save_path = "{}-segmented.png".format(impath.stem)
        utils.save_im(save_path, segmented_image)
