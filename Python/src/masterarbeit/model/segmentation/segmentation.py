from abc import ABCMeta
from abc import abstractmethod
from masterarbeit.model.segmentation.helpers import mask, crop, simple_binarize

class Segmentation(metaclass=ABCMeta):  
    
    label = 'None'   
    
    def process(self, image, steps=None):
        binary = self._process(image, steps=steps)
        cropped = crop(mask(image, binary))
        return cropped
         
    @abstractmethod
    def _process(self, image, steps=None):
        return