import argparse
import functools
import os
from src.color_conversion import cv2_grey_to_color
from src.image_preprocess import image_preparation
import SimpleITK as sitk


def directory(dir_name):
    if not os.path.isfile(dir_name):
        raise argparse.ArgumentTypeError(dir_name + " is not a valid directory name")
    return dir_name


def get_args():
    """
    Constructs argument parser and parse the arguments
    """
    parser = argparse.ArgumentParser(description="Perform image comparison using ssim.")
    parser.add_argument("--original_path", type=directory, help="original file.")
    parser.add_argument(
        "--contrasted_path", type=directory, help="contrast enhanced file."
    )
    parser.add_argument(
        "--sephiaed_path", type=directory, help="sephiaed enhance file."
    )

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


if __name__ == "__main__":
    args = get_args()
    pass
