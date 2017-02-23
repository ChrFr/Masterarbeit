import numpy as np
from abc import ABCMeta
from abc import abstractmethod
import math
import cv2
import uuid
from masterarbeit.model.segmentation.helpers import simple_binarize

class Feature(metaclass=ABCMeta):   
    label = 'None'
    columns = None      
    binary_input = False
    n_pixels = 2000000
    
    def __init__(self, category, id=None):
        if id is None:
            id = str(uuid.uuid4())
        self.id = id
        self._v = None
        self.category = category
        
    def _common_scale(self, image):
        resolution = image.shape[0] * image.shape[1]
        scale = math.sqrt(self.n_pixels / resolution)        
        new_shape = (np.array(image.shape[:2]) * scale).astype(np.int)
        # cv2 swaps width with height
        scaled = cv2.resize(image, (new_shape[1], new_shape[0]))
        return scaled    
    
    @property
    def is_described(self):
        return self._v is not None
    
    @property    
    def values(self):
        if self._v is None:
            raise Exception('Feature has not been described yet!')
        return self._v               
    
    @values.setter    
    def values(self, values):
        values = values.flatten()     
        self._v = values
        
    def describe(self, segmented_image, steps=None):
        if self.binary_input:
            segmented_image = simple_binarize(segmented_image)
        self._v = self._describe(segmented_image, steps=steps)
        if self._v is None:
            return False
        return True
    
    @abstractmethod
    def _describe(self, image, steps=None):
        pass
    
    
class UnsupervisedFeature(Feature):
    n_levels = 1
    histogram_length = 50
    
    @property
    def data_is_analysed(self):
        return (self.is_described and 
                len(self._v.shape) == 1 and 
                type(self._v) != np.ndarray)      
        
    def __init__(self, category, id=None):
        super(UnsupervisedFeature, self).__init__(category, id=id)
        
    def transform(self, codebook):
        if not isinstance(codebook, self.codebook_type):
            raise Exception('Feature requires {} to build an histogram'
                            .format(self.codebook_type.__name__))
        self._v = codebook.histogram(self.values) 
                

class JoinedFeature(Feature):
    label = 'Joined Feature'
    
    def __init__(self, category, id=None):
        super(JoinedFeature, self).__init__(category, id=id)
        self._v = np.empty(0)
    
    def add(self, values):
        self._v = np.concatenate((self._v, values))
        
    def describe(self, image, steps=None):
        pass    
    
    def _describe(self, image, steps=None):
        pass    