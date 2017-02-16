import numpy as np
from abc import ABCMeta
from abc import abstractmethod
import math
import cv2

class Feature(metaclass=ABCMeta):   
    label = 'None'    
    columns = None      
    binary_input = True
    common_resolution = 2000000
    
    def __init__(self, category):
        self._v = None
        self.category = category
        
    def _common_scale(self, image):
        resolution = image.shape[0] * image.shape[1]
        scale = math.sqrt(self.common_resolution / resolution)        
        new_shape = (np.array(image.shape[:2]) * scale).astype(np.int)
        # cv2 swaps width with height
        scaled = cv2.resize(image, (new_shape[1], new_shape[0]))
        return scaled
        
    @property    
    def values(self):
        if self._v is None:
            raise Exception('Feature has not been described yet!')
        return self._v               
    
    @values.setter    
    def values(self, values):
        values = values.flatten()        
        if self.columns is not None and len(values) != len(self.columns):
            raise Exception('expected length of feature mismatches ' +
                            'length of extracted values! ({}!={})'.format(
                                len(self.columns), len(values)
                            ))
        self._v = values
        
    def describe(self, image, steps=None):
        self._v = self._describe(image, steps=steps)
        if self._v is None:
            return False
        return True
    
    @abstractmethod
    def _describe(self, image, steps={}):
        pass
    
class UnsupervisedFeature(Feature):
    n_levels = 1
    codebook_type = None
    histogram_length = 50
    
    @classmethod
    def new_codebook(cls):
        codebook = cls.codebook_type(cls.histogram_length, 
                                     n_levels=cls.n_levels)
        return codebook
        
    def __init__(self, category):
        super(UnsupervisedFeature, self).__init__(category)
        
    def histogram(self, codebook):
        if not isinstance(codebook, self.codebook_type):
            raise Exception('Feature requires {} to build an histogram'
                            .format(self.codebook_type.__name__))
        self._v = codebook.histogram(self.values) 
                