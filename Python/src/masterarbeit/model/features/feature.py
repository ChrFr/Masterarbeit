import numpy as np
from abc import ABCMeta
from abc import abstractmethod

class Feature():   
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
    
class MultiFeature(Feature):
    dictionary_type = None
        
    def __init__(self, category):
        super(MultiFeature, self).__init__(category)
        self._raw = None
        
    @property    
    def values(self):
        if self._raw is None:
            raise Exception('Feature has not been described yet!')
        if self._v is None:
            raise Exception("Histogram wasn't built yet!")
        return self._v       
    
    def build_histogram(self, dictionary):
        if not isinstance(dictionary, self.dictionary_type):
            raise Exception('Feature requires {} to build an histogram'.format(
                self.dictionary_type.__name__                
            ))
        if len(dictionary) != len(self):
            raise Exception('Length of atoms mismatch! ({}!={})i'.format(
                len(dictionary), len(self)               
            ))        
        self._v = self.get_atom_histogram(dictionary)        
    
    def get_atom_histogram(self, dictionary):        
        feature_vector = self._raw.reshape(self._raw.shape[0], -1) 
        histogram = dictionary.count_patterns(feature_vector)
        return histogram