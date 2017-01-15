import numpy as np
from collections import OrderedDict
import os
from skimage import color
from skimage import io
from skimage.filters import threshold_otsu

from masterarbeit.model.preprocessor.preprocessor import PreProcessor

class SkimagePreprocessor(PreProcessor):
    
    @staticmethod
    def read(filename):    
        pixel_array = io.imread(filename)
        return pixel_array
    
    @staticmethod
    def write(filename):    
        pass
    
class BinarizeHSV(SkimagePreprocessor): 
    
    @staticmethod
    def process(image, steps_dict=None):
        hsv = color.rgb2hsv(image)   
        steps_dict['hsv'] = hsv    
        s = hsv[:,:,1]
        threshold_global_otsu = threshold_otsu(s)
        global_otsu = s <= threshold_global_otsu
        
        steps_dict['i'] = image        
        steps_dict['otsu'] = global_otsu
        binary = np.zeros(global_otsu.shape, dtype='uint8')
        binary[-global_otsu] = 255
        steps_dict['binary'] = binary
        #from matplotlib import pyplot as plt
        #fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True,
                               #figsize=(6, 8))
        
        #ax[0].imshow(image)
        #ax[1].imshow(binary)
        #plt.show()
        return binary
    