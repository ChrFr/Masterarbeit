from abc import ABCMeta
from abc import abstractmethod

class Segmentation(metaclass=ABCMeta):  
    
    label = 'None'    
         
    @abstractmethod
    def process(self, image, steps=None):
        return