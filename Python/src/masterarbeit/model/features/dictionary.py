from sklearn.decomposition import MiniBatchDictionaryLearning, SparseCoder
import numpy as np

class MiniBatchDictionary():     
    n_atoms = 50    
    
    def __init__(self):
        self.atoms = None
        self.coder = None
        self.dictionary = None
        
    def build(self, features):        
        feature_vector = features.reshape(features.shape[0], -1)
        minibatch = MiniBatchDictionaryLearning(n_components=self.n_atoms, 
                                                alpha=1, n_iter=500)
        self.dictionary = minibatch.fit(feature_vector)
        self.atoms = self.dictionary.components_
        
    def __len__(self):
        return self.n_atoms
        
    def count_patterns(self, feature_vector):
        d = self.dictionary if self.dictionary is not None else self.coder
        if d is None:
            raise Exception('no dictionary or coder found')
        b = d.transform(feature_vector)
        histogram = []
        # count, how often pattern appears
        for i in range(b.shape[1]):
            histogram.append(np.count_nonzero(b[:, i]))
        return histogram
            
    def load(self, atoms):
        self.atoms = atoms
        # use coder instead of dictionary, performance wise
        self.coder = SparseCoder(atoms)             
        
    #def update(self, features):
        #self.dictionary.partial_fit(a)        
        
        
class PyramidMiniBatchDictionary():
    
    def get_dict(self, level):
        return self.levels[level]