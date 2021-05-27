# USAGE
# python color_conversion.py --image ../img/Image_1.png --ouput ./
# Author: Miguel Rueda
# e-mail: makquel@gmail.com

# FIXME: Check another package for NMS
# from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import functools
import cv2
import json
import time
import os


def directory(dir_name):
    if not os.path.isfile(dir_name):
        raise argparse.ArgumentTypeError(dir_name + "is not a valid directory name")
    return dir_name


def get_args():
    """
    Constructs argument parser and parse the arguments
    """
    parser = argparse.ArgumentParser(description="Perform Template Matching.")
    parser.add_argument("--sample_path", type=directory, help="sample file.")
    parser.add_argument("--template_path", type=directory, help="template file.")

    return parser.parse_args()


def timer(func):
    """
    Prints out the runtime of the decorated function
    """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def GrayScaleToBlueToRedColor(intensity, norm_value):
    """
    intensity: pixel intensity value
    norm_value: max value (e.g. 2^8-1)
    """
    value = 4.0 * (float(intensity) / float(norm_value)) + 1

    return (
        norm_value * np.max([0.0, (3.0 - abs(value - 4) - abs(value - 5)) / 2]),
        norm_value * np.max([0.0, (4.0 - abs(value - 2) - abs(value - 4)) / 2]),
        norm_value * np.max([0.0, (3.0 - abs(value - 1) - abs(value - 2)) / 2]),
    )


def GrayImageToColorImage(image) -> np.ndarray:
    """
    image:grayscale image
    """
    colored_image = np.zeros([image.shape[0], image.shape[1], 3])
    Imax = np.max(image.ravel())

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            sRGB = GrayScaleToBlueToRedColor(image[i, j], Imax)
            colored_image[i, j, 2] = np.floor(sRGB[0]).astype(int)
            colored_image[i, j, 1] = np.floor(sRGB[1]).astype(int)
            colored_image[i, j, 0] = np.floor(sRGB[2]).astype(int)
    # FIXME:
    #     return cv2.cvtColor(colored_image, cv2.COLOR_BGR2YCrCb)
    return colored_image


def get_args():
    """
    construct the argument parser and parse the arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-img", "--image", type=str, help="path to input image")
    # parser.add_argument("-out", "--output", type=str, help="path to output")

    # parser.add_argument("-ig", "--image-gray", type=str, default="../img/Image_1.png", help="path to input image")

    return parser.parse_args()


def cv2_grey_to_color(image) -> np.ndarray:
    r"""Converts an image array from grayscale (3 stacked channels) to YCbCr

    Parameters
    ----------
    image : OpenCV image object
        uint8 tuple-like (nx, ny, 3)
    Returns
    -------
    type:
        numpy ndarray object.
    describe :
        pseudo-colored image using YCbCr color space
    """
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    R_channel = []
    G_channel = []
    B_channel = []
    ## Create LUT Red-Blue table
    H = pow(2, 8)
    for elt in range(0, H):
        R, G, B = np.floor(GrayScaleToBlueToRedColor(elt, H - 1)).astype("uint8")
        R_channel.append(R)
        G_channel.append(G)
        B_channel.append(B)

    R_channel = np.asarray(R_channel)
    G_channel = np.asarray(G_channel)
    B_channel = np.asarray(B_channel)

    lut = np.dstack((B_channel, G_channel, R_channel))
    image = cv2.LUT(image, lut)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    return image


# FIXME: Create unit test
# if __name__ == '__main__':

# 	args = get_args();

# 	start_time = time.time()

# 	image = cv2.imread( args.image )
# 	orig = image.copy()
# 	# image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# 	R_channel = []
# 	G_channel = []
# 	B_channel = []
# 	## Create LUT Red-Blue table
# 	H = pow(2,8)
# 	for elt in range(0,H):
# 	    # lut_x = np.append(lut_x, np.floor(GrayScaleToBlueToRedColor(elt,255)).astype('uint8'), axis=0)
# 	    R,G,B = np.floor(GrayScaleToBlueToRedColor(elt,H-1)).astype('uint8')
# 	    R_channel.append(R)
# 	    G_channel.append(G)
# 	    B_channel.append(B)

# 	R_channel = np.asarray(R_channel)
# 	G_channel = np.asarray(G_channel)
# 	B_channel = np.asarray(B_channel)

# 	lut = np.dstack((B_channel, G_channel, R_channel))
# 	print(image.shape)
# 	print(lut.shape)
# 	image = cv2.LUT(image, lut)
# 	image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# 	cv2.imwrite('./pseudo_colored.png', image)

# 	elapsed_time = time.time() - start_time
# 	print(elapsed_time)
