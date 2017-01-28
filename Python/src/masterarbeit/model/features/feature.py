import numpy as np
from abc import ABCMeta
from abc import abstractmethod

class Feature():   
    label = 'None'    
    columns = []    
    category = None
    
    def __init__(self, category):
        self._v = np.empty(self.length) 
        self.category = category
        
    @property
    def length(self):
        return len(self.columns)
    
    @property    
    def values(self):
        return self._v    
    
    @values.setter    
    def values(self, values):
        values = values.flatten()
        if len(values) != self.length:
            raise Exception('expected length of feature mismatches ' +
                            'length of extracted values! ({}!={})'.format(
                                self.length, len(values)
                            ))
        self._v = values
        
    @abstractmethod
    def extract(self, image):
        pass