import utils
import numpy as np


def _extend(im, x_src, y_src):
    x_res, y_res = im.shape
    points = []
    for x in range(max(0, x_src - 1), min(x_res, x_src + 2)):
        for y in range(max(0, y_src - 1), min(y_res, y_src + 2)):
            if x == x_src and y == y_src:
                continue
            points.append((x, y))
    return np.array(points)


def region_growing(im: np.ndarray, seed_points: list, T: int) -> np.ndarray:
    """
    A region growing algorithm that segments an image into 1 or 0 (True or False).
    Finds candidate pixels with a Moore-neighborhood (8-connectedness).
    Uses pixel intensity thresholding with the threshold T as the homogeneity criteria.
    The function takes in a grayscale image and outputs a boolean image

    args:
        im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
        seed_points: list of list containing seed points (row, col). Ex:
            [[row1, col1], [row2, col2], ...]
        T: integer value defining the threshold to used for the homogeneity criteria.
    return:
        (np.ndarray) of shape (H, W). dtype=np.bool
    """
    # START YOUR CODE HERE ### (You can change anything inside this block)
    # You can also define other helper functions
    segmented = np.zeros_like(im).astype(bool)
    im = im.astype(float)

    def check_homogenicity(p1, p2):
        return np.abs(im[p1[0], p1[1]] - im[p2[0], p2[1]]) < T

    for seed in seed_points:
        queue = [seed]

        while queue:
            p = queue.pop()
            # Avoid double checking pixels
            if segmented[p[0], p[1]]:
                continue

            segmented[p[0], p[1]] = True
            adjacent = _extend(im, *p)
            for adj in adjacent:
                if check_homogenicity(seed, adj):
                    queue.append(adj)

    return segmented
    ### END YOUR CODE HERE ###


if __name__ == "__main__":
    # DO NOT CHANGE
    im = utils.read_image("defective-weld.png")

    seed_points = [  # (row, column)
        [254, 138],  # Seed point 1
        [253, 296],  # Seed point 2
        [233, 436],  # Seed point 3
        [232, 417],  # Seed point 4
    ]
    intensity_threshold = 50
    segmented_image = region_growing(im, seed_points, intensity_threshold)

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
    utils.save_im("defective-weld-segmented.png", segmented_image)
