import numpy as np
from abc import ABCMeta
from abc import abstractmethod

class Feature(metaclass=ABCMeta):   
    label = 'None'    
    columns = []      
    
    def __init__(self, category):
        self._v = None
        self.category = category
    
    def __len__(self):
        return len(self.columns)
    
    @property    
    def values(self):
        if self._v is None:
            raise Exception('Feature has not been described yet!')
        return self._v               
    
    @values.setter    
    def values(self, values):
        values = values.flatten()
        if len(values) != len(self):
            raise Exception('expected length of feature mismatches ' +
                            'length of extracted values! ({}!={})'.format(
                                len(self), len(values)
                            ))
        self._v = values
        
    @abstractmethod
    def describe(self, image, steps={}):
        pass
    
class UnsupervisedFeature(Feature):
    n_levels = 1
    codebook_type = None     
    
    @staticmethod
    def new_codebook():
        codebook = self.codebook_type(self.n_levels)
        
    def __init__(self, category):
        super(UnsupervisedFeature, self).__init__(category)
        
    def histogram(self, codebook):
        values = self.values
        if not isinstance(codebook, self.codebook_type):
            raise Exception('Feature requires {} to build an histogram'
                            .format(self.codebook_type.__name__))
        if len(codebook) != len(self):
            raise Exception('Length of atoms mismatch! ({}!={})i'.format(
                len(codebook), len(self)               
            ))        
        feature_vector = values.reshape(values.shape[0], -1) 
        self._v = codebook.histogram(feature_vector) 
                