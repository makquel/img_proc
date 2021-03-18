# USAGE

# Author: Miguel Rueda
# e-mail: makquel@gmail.com


import numpy as np
import SimpleITK as sitk
import gc
import sys
import os
import logging
logger = logging.getLogger('pre_process')
SRC_PATH = os.path.dirname(os.path.abspath(os.getcwd())) #os.getenv('SRC_PATH')
sys.path.insert(1, SRC_PATH)
from color_conversion import cv2_grey_to_color

class image_preparation():
    '''
    Class for pre-processing CT(Computed tomography) images.
    '''
    def __init__(self,
                 image_path,
                 window_level    = -600,
                 window_width    = 1500,
                 cormack_level   = False,
                 image_window    = True,
                 image_size = [256,256,256]
                 ):

        self.raw_image = image_path
        self.window_level = window_level
        self.window_width = window_width
        self.cormack_level = cormack_level
        if self.cormack_level:
            # hardcoded cormack level
            self.window_level = 0
            self.window_width = 2000

        self.image_window = image_window
        self.image_size = image_size
        
        try:
            self.image = self.loadImage()
        except:
            logger.error('File {} not found or not supported!'.format(self.raw_image))

    # TODO: Check wether this method is necessary 
    def __del__(self):
        logger.info("Imag pre-process destructor Called!") 
        

    ## TODO: use this for simplifing purpouse
    @staticmethod
    def _prepare_size(image_path, downsample_rate): 
        Size = image_path.GetSize()/downsample_rate
        return Size

    def getImage(self):
        '''
        Returns the image in the SimpleITK object format.
        '''
        return self.image
   
    def setWindowLevel(self, value):
        '''
        Sets the minimum window value.
        '''
        self.window_level = value
   
    def setWindowWidth(self, value):
        '''
        Sets the maximum window value.
        '''
        self.window_width = value
        
    def setScaleCormack(self, flag):
        '''
        Sets the Boolean value indicating whether or not the Hounsfield scale conversion to Cormack level
        will be performed.
        '''
        self.cormack_level = flag
    
    def setImageWindow(self, flag):
        '''
        Sets the Boolean value indicating whether or not the image will be windowed.
        '''
        self.image_window = flag
  
    def hounsfield_to_cormack(self, image):
        '''
        Conversion formula suggested by Chris Rorden
        in matlab's clinical toolbox
        https://www.nitrc.org/projects/clinicaltbx/
        '''
        img_data = sitk.GetArrayFromImage(image)
        t = img_data.flatten()
        t1 = np.zeros(t.size)
        t1[np.where(t>100)] = t[np.where(t > 100)]+3000
        t1[np.where(np.logical_and(t >= -1000, t <= -100))]=t[np.where(np.logical_and(t >= -1000,t <= -100))]+1000
        t1[np.where(np.logical_and(t >= -99, t <= 100))]=(t[np.where(np.logical_and(t >= -99, t <= 100))]+99)*11+911
        trans_img = t1.reshape(img_data.shape)

        res_img = sitk.GetImageFromArray(trans_img)
        res_img.CopyInformation(image)

        return res_img
    
    def imageWindow(self, image):
        '''

        '''
        if self.cormack_level:
            image = self.hounsfield_to_cormack(image)
            logger.info("Cormack level selected")

        windowing = sitk.IntensityWindowingImageFilter()
        windowing.SetWindowMinimum(self.window_level)
        windowing.SetWindowMaximum(self.window_width)
        img_win = windowing.Execute(image)

        return img_win
        
    def loadImage(self):
        '''
 
        '''
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
    
    def downsampleImage(self,image):
        '''
        Downsample funtion for deep learning preprocess purpose
        image: 
        reference_size: downsampled size in vector like format (i.e. [sx, sy, sz])
        '''
        #TODO: add loggin capabilities
        original_CT = image
        # NIfTi(RAS) to ITK(LPS) 
        original_CT = sitk.DICOMOrient(original_CT, 'LPS')
        dimension = original_CT.GetDimension()
        reference_physical_size = np.zeros(original_CT.GetDimension())
        reference_physical_size[:] = [(sz-1)*spc if sz*spc>max_  else max_ for sz,spc,max_ in zip(original_CT.GetSize(), original_CT.GetSpacing(), reference_physical_size)]
        
        reference_origin = original_CT.GetOrigin()
        reference_direction = original_CT.GetDirection()
        #FIXME: Looks like the downsampled image is mirrored over the y axis
    #     reference_direction = [1.,0.,0.,0.,1.,0.,0.,0.,1.]
        reference_size = self.image_size
        logger.info(str(reference_size))
        reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

        reference_image = sitk.Image(reference_size, original_CT.GetPixelIDValue())
        reference_image.SetOrigin(reference_origin)
        reference_image.SetSpacing(reference_spacing)
        reference_image.SetDirection(reference_direction)

        reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
        
        transform = sitk.AffineTransform(dimension)
        transform.SetMatrix(original_CT.GetDirection())
        # transform.SetMatrix([1,0,0,0,-1,0,0,0,1])
        transform.SetTranslation(np.array(original_CT.GetOrigin()) - reference_origin)
        # Modify the transformation to align the centers of the original and reference image instead of their origins.
        centering_transform = sitk.TranslationTransform(dimension)
        img_center = np.array(original_CT.TransformContinuousIndexToPhysicalPoint(np.array(original_CT.GetSize())/2.0))
        centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
        centered_transform = sitk.CompositeTransform([transform, centering_transform])

        # sitk.Show(sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0))
        
        return sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0)



    def gantryRemoval(self):
        '''

        '''
        #FIXME: Merge with class for full funtinolaty
        #TODO: Just a MWE of ConnectedComponent function
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
        chestMask = sitk.BinaryMorphologicalOpening(cc, [2]*3)  
        # apply mask on image and return
        tmp_img = sitk.Mask(ct_image, chestMask, min_hu) 

        return tmp_img

    
