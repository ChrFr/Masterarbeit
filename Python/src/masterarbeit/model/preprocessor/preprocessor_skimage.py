import numpy as np
from collections import OrderedDict
import os
from skimage import color
from skimage import io
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, dilation, remove_small_objects
from skimage.morphology import disk, square
from skimage.transform import resize
from masterarbeit.model.preprocessor.common import mask

from masterarbeit.model.preprocessor.preprocessor import PreProcessor

class SkimagePreprocessor(PreProcessor):
    process_width = 1280
    
    @staticmethod
    def read(filename):    
        pixel_array = io.imread(filename)
        return pixel_array
    
    @staticmethod
    def write(image, filename):
        try:
            io.imsave(filename, image)
        # possible exceptions: file not found, no permission, 
        # malformed image array (not specifically handled here)
        except: 
            return False
        return True        
    
    @staticmethod
    def scale(image, shape):      
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
    
class BinarizeHSV(SkimagePreprocessor): 
    
    
    @staticmethod
    def fill_holes(binary, size, iterations=3):     
        for i in range(iterations):
            dilated = dilation(binary, selem=square(size)) 
        return dilated
    
    @staticmethod
    def remove_stem(binary):   
        # you have to fill the little holes otherwise the stem removal 
        # reacts on them in an undesired way 
        # (big kernel messes up contour, so only as step for stem removal)
        wo_holes = __class__.fill_holes(binary, 7)
        # remove stem by eroding small structures
        for i in range(3):
            eroded = erosion(wo_holes, selem=disk(12))
        # dilate with bigger structuring element ('blow it up'), 
        # because relevant details might also got lost in erosion
        for i in range(3):
            dilated = dilation(eroded, selem=disk(15))
        # sometimes the base of the stem remains -> remove it
        rem = remove_small_objects(dilated.astype(bool), min_size=10000)
        rem = rem.astype(np.uint8)
        return rem
    
    @staticmethod
    def binarize_otsu(axis):        
        threshold = threshold_otsu(axis)
        otsu = axis <= threshold
        binary = np.zeros(otsu.shape, dtype=np.uint8)
        binary[-otsu] = 1
        return binary   
    
    @staticmethod
    def process(image, steps_dict=None):
        o_height = image.shape[0]
        o_width = image.shape[1]
        res_factor = __class__.process_width / o_width
        new_shape = (int(o_height * res_factor), int(o_width * res_factor))
        resize = __class__.scale(image, new_shape)
        
        hsv = color.rgb2hsv(resize)          
        s = hsv[:,:,1]
        res_binary = __class__.binarize_otsu(s) 
        wo_stem = __class__.remove_stem(res_binary)
        
        # fill small holes in final binary mask (smaller kernel to keep contour)
        #wo_holes = __class__.fill_holes(res_binary, 5, iterations=2)         
        masked_binary = mask(res_binary, wo_stem)       
        # rescale
        masked_binary = __class__.scale(masked_binary, (o_height, o_width))  
        
        if steps_dict is not None:
            steps_dict['hsv'] = hsv
            steps_dict['binary (resized)'] = res_binary
            steps_dict['without stem (resized)'] = wo_stem 
            steps_dict['binary mask without stem'] = masked_binary         
        
        return masked_binary        