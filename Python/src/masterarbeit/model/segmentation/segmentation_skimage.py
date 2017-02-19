import numpy as np
from collections import OrderedDict
import os
from skimage import color
from skimage import io
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, dilation, remove_small_objects
from skimage.morphology import disk, square
from skimage.transform import resize
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel

from masterarbeit.model.segmentation.segmentation import Segmentation
    
    
class BinarizeHSV(Segmentation):     
    
    label = 'Binarize image with Otsu-Thresholding in HSV space (Skimage)' 
    process_width = 2000
        
    def binarize_otsu(self, axis):        
        threshold = threshold_otsu(axis)
        otsu = axis <= threshold
        binary = np.zeros(otsu.shape, dtype=np.uint8)
        binary[-otsu] = 1
        return binary   
    
    def scale(self, image, shape):      
        '''
        resize image to given shape
        '''
        shape = list(shape)
        for i in range(2, len(image.shape)):
            shape.append(image.shape[i])
        resized = resize(image.copy(), shape, preserve_range=True)
        # if image was binary -> back to binary range [0,1] by rounding
        # (scaling interpolates)
        if ((image==0) | (image==1)).all():  
            resized = np.rint(resized)
        # back to original dtype (scaling returns floats)
        resized = resized.astype(image.dtype)
        return resized    
    
    def _process(self, image, steps=None):
        # scale image to a uniform size, so common structured elements can be used
        # (and to reduce calc. time)
        o_height = image.shape[0]
        o_width = image.shape[1]
        res_factor = self.process_width / o_width
        new_shape = (int(o_height * res_factor), int(o_width * res_factor))
        resize = self.scale(image, new_shape)
        
        hsv = color.rgb2hsv(resize)      
        # take values only
        s = hsv[:,:,1]
        res_binary = self.binarize_otsu(s) 
        # remove holes in leaf (caused by reflections or actual holes)
        wo_holes = (ndi.binary_fill_holes(
            res_binary,structure=np.ones((5,5)))).astype(np.uint8)
        # rescale
        binary = self.scale(wo_holes, (o_height, o_width))  
        
        #TODO: maybe do removal of small holes on masked_binary instead of stem (remove fragment border pixels)
        
        if steps is not None:
            steps['hsv'] = hsv
            steps['binary (resized)'] = res_binary
            steps['binary without holes (resized)'] = wo_holes
        
        return binary        
    