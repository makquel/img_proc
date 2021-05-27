""" USAGE

Author: Miguel Rueda
e-mail: makquel@gmail.com

"""

import numpy as np
import SimpleITK as sitk
import gc
import sys
import os
import logging
from .color_conversion import cv2_grey_to_color
from dataclasses import dataclass

logger = logging.getLogger("pre_process")


@dataclass
class image_preparation:
    r""" """

    def __init__(
        self,
        image_path,
        window_level=-600,
        window_width=1500,
        cormack_level=False,
        image_window=True,
        image_size=[256, 256, 256],
    ):

        self.raw_image = image_path
        self.window_level = window_level
        self.window_width = window_width
        self.cormack_level = cormack_level
        if self.cormack_level:
            self.window_level = 0
            self.window_width = 2000

        self.image_window = image_window
        self.image_size = image_size

        try:
            self.image = self.loadImage()
        except:
            logger.error("File {} not found or not supported!".format(self.raw_image))

    # TODO: Check whether this method is necessary
    def __del__(self):
        logger.info("Image pre-process destructor Called!")

    ## TODO: use this for simplifing purpouse
    @staticmethod
    def _prepare_size(image_path, downsample_rate):
        Size = image_path.GetSize() / downsample_rate
        return Size

    def getImage(self):
        """
        Returns the image in the SimpleITK object format.
        """
        return self.image

    def setWindowLevel(self, value):
        """
        Sets the minimum window value.
        """
        self.window_level = value

    def setWindowWidth(self, value):
        """
        Sets the maximum window value.
        """
        self.window_width = value

    def setScaleCormack(self, flag):
        """
        Sets the Boolean value indicating whether or not the Hounsfield scale
        conversion to Cormack level will be performed.
        """
        self.cormack_level = flag

    def setImageWindow(self, flag):
        """
        Sets the Boolean value indicating whether or not the image will be
        windowed.
        """
        self.image_window = flag

    def hounsfield_to_cormack(self, image) -> sitk.Image:
        r"""Conversion formula suggested by Chris Rorden
        in matlab's clinical toolbox
        https://www.nitrc.org/projects/clinicaltbx/

        Parameters
        ----------
        image : SimpleITK image object
            Raw CT scan (Hounsfield scale)
        Returns
        -------
        type:
            SimpleITK image object.
        describe :
            Cormack scaled CT scan
        """
        img_data = sitk.GetArrayFromImage(image)
        t = img_data.flatten()
        t1 = np.zeros(t.size)
        t1[np.where(t > 100)] = t[np.where(t > 100)] + 3000
        t1[np.where(np.logical_and(t >= -1000, t <= -100))] = (
            t[np.where(np.logical_and(t >= -1000, t <= -100))] + 1000
        )
        t1[np.where(np.logical_and(t >= -99, t <= 100))] = (
            t[np.where(np.logical_and(t >= -99, t <= 100))] + 99
        ) * 11 + 911
        trans_img = t1.reshape(img_data.shape)

        res_img = sitk.GetImageFromArray(trans_img)
        res_img.CopyInformation(image)

        return res_img

    def imageWindow(self, image) -> sitk.Image:
        r"""A one-line summary that does not use variable names or the
        function name.
        Several sentences providing an extended description. Refer to
        variables using back-ticks, e.g. `var`.

        Parameters
        ----------
        image : array_like
            Array_like means all those objects -- lists, nested lists, etc. --
            that can be converted to an array.  We can also refer to
            variables like `var1`.
        Returns
        -------
        type
            Explanation of anonymous return value of type ``type``.
        describe : type
            Explanation of return value named `describe`.
        out : type
            Explanation of `out`.
        Other Parameters
        ----------------
        only_seldom_used_keywords : type
            Explanation
        common_parameters_listed_above : type
            Explanation
        """
        if self.cormack_level:
            image = self.hounsfield_to_cormack(image)
            logger.info("Cormack level selected")

        windowing = sitk.IntensityWindowingImageFilter()
        windowing.SetWindowMinimum(self.window_level)
        windowing.SetWindowMaximum(self.window_width)
        img_win = windowing.Execute(image)

        return img_win

    def loadImage(self) -> sitk.Image:
        """ """
        # Reads the image using SimpleITK
        image = sitk.ReadImage(self.raw_image, imageIO="NiftiImageIO")
        # Performs linear windowing
        if self.image_window:
            ct_image = self.imageWindow(image)
        else:
            ct_image = image
        # ct_array = sitk.GetArrayFromImage(itkimage)
        # Performs downsampling for Deep Learning purpouse
        ct_image = self.downsampleImage(ct_image)
        logger.info(str(ct_image.GetSize()))

        return ct_image

    def downsampleImage(self, image) -> sitk.Image:
        r"""Downsample funtion for deep learning preprocess purpose image
        reference_size: size in vector like format (i.e. [sx, sy,sz])
        Parameters
        ----------
        image : SimpleITK image object
            Windowed CT scan
        Returns
        -------
        type:
            SimpleITK image object.
        describe :
            Downsampled CT scan
        """
        original_CT = image
        # NIfTi(RAS) to ITK(LPS)
        original_CT = sitk.DICOMOrient(original_CT, "LPS")
        dimension = original_CT.GetDimension()
        reference_physical_size = np.zeros(original_CT.GetDimension())
        reference_physical_size[:] = [
            (sz - 1) * spc if sz * spc > max_ else max_
            for sz, spc, max_ in zip(
                original_CT.GetSize(), original_CT.GetSpacing(), reference_physical_size
            )
        ]

        reference_origin = original_CT.GetOrigin()
        reference_direction = original_CT.GetDirection()
        # NOTE: Looks like the downsampled image is mirrored over the y axis
        #     reference_direction = [1.,0.,0.,0.,1.,0.,0.,0.,1.]
        reference_size = self.image_size
        logger.info(str(reference_size))
        reference_spacing = [
            phys_sz / (sz - 1)
            for sz, phys_sz in zip(reference_size, reference_physical_size)
        ]

        reference_image = sitk.Image(reference_size, original_CT.GetPixelIDValue())
        reference_image.SetOrigin(reference_origin)
        reference_image.SetSpacing(reference_spacing)
        reference_image.SetDirection(reference_direction)

        reference_center = np.array(
            reference_image.TransformContinuousIndexToPhysicalPoint(
                np.array(reference_image.GetSize()) / 2.0
            )
        )

        transform = sitk.AffineTransform(dimension)
        transform.SetMatrix(original_CT.GetDirection())
        transform.SetTranslation(np.array(original_CT.GetOrigin()) - reference_origin)
        # Modify the transformation to align the centers of the original and
        # reference image instead of their origins.
        centering_transform = sitk.TranslationTransform(dimension)
        img_center = np.array(
            original_CT.TransformContinuousIndexToPhysicalPoint(
                np.array(original_CT.GetSize()) / 2.0
            )
        )
        centering_transform.SetOffset(
            np.array(
                transform.GetInverse().TransformPoint(img_center) - reference_center
            )
        )
        centered_transform = sitk.CompositeTransform([transform, centering_transform])

        return sitk.Resample(
            original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0
        )

    def gantryRemoval(self, ct_image) -> sitk.Image:
        r"""Intensity based tight crop

        Parameters
        ----------
        ct_image : SimpleITK image object
            Raw CT scan (Hounsfield scale)
        Returns
        -------
        type:
            SimpleITK image object.
        describe :
            tight cropped CT scan
        """
        max_hu = int(np.max(sitk.GetArrayFromImage(ct_image)))
        min_hu = int(np.min(sitk.GetArrayFromImage(ct_image)))
        # binary mask of chest + artifact(s)
        image_thr = sitk.BinaryThreshold(ct_image, -600, max_hu)
        # label image with connected components
        cc = sitk.ConnectedComponent(image_thr)
        # reorder labels with largest components first
        cc = sitk.RelabelComponent(cc)
        # get the first component (chest)
        cc = sitk.BinaryThreshold(cc, 1, 1)
        #  erode and dilate to denoise
        chestMask = sitk.BinaryMorphologicalOpening(cc, [2] * 3)
        # apply mask on image and return
        tmp_img = sitk.Mask(ct_image, chestMask, min_hu)

        return tmp_img
