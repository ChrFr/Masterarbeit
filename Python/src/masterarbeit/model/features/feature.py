import numpy as np

class Feature():   
    def __init__(self, name):
        self.name = name
        self._v = np.empty(0) 
        
    @property
    def shape(self):
        return self._v.shape
    
    @property    
    def values(self):
        return self._v    
    
    @values.setter    
    def values(self, values):
        self._v = values
