from sklearn.decomposition import (MiniBatchDictionaryLearning, SparseCoder,
                                   PCA, DictionaryLearning)
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import numpy as np
import pickle

from abc import ABCMeta, abstractmethod

class Codebook(metaclass=ABCMeta):  
    n_components = 0
    n_levels = 1
    
    def __init__(self, levels=1):
        self.n_levels = levels
        self.codebooks = []
        
    def fit(self, feature_values):           
        self.codebooks = []
        if self.n_levels <= 1:
            feature_values = [feature_values]
        for level in feature_values:       
            codebook = self._fit(level)
            self.codebooks.append(codebook)
      
    def histogram(self, feature_values):        
        if self.n_levels <= 1:
            feature_values = [feature_values]        
        histogram = np.array([])
        # concatenate all histograms
        for i, codebook in enumerate(self.codebooks):            
            histogram = np.concatenate(
                (histogram, self._histogram(codebook, feature_values[i])))
        return histogram
        
    def deserialize(self, serialized):
        if self.n_levels <= 1:
            serialized = [serialized]
        for i in range(self.n_levels):
            self.codebooks.append(self._deserialize(serialized[i]))
    
    def serialize(self): 
        serialized = []
        for codebook in self.codebooks:  
            serialized.append(self._serialize())
        return None    
    
    @abstractmethod
    def _fit(self, feature_values):
        pass
    
    @abstractmethod    
    def _histogram(self, codebook, feature_vector):
        pass
        
    @abstractmethod 
    def _deserialize(self, serialized):
        pass
    
    @abstractmethod 
    def _serialize(self):
        return None    
    
    def __len__(self):
        return self.n_components    

class DictLearningCodebook(Codebook):     
    n_components = 100    
    
    def __init__(self):
        super(DictLearningCodebook, self).__init__()
        self.components = None
        self.coder = None
        
    def _fit(self, feature_values):   
        # reshape to two dimensional vector (list of 1d features)
        feature_vector = feature_values.reshape(feature_values.shape[0], -1)         
        minibatch = MiniBatchDictionaryLearning(n_components=self.n_components, 
                                                alpha=1, n_iter=500)
        codebook = minibatch.fit(feature_vector)
        return codebook     
        
    def _histogram(self, codebook, feature_vector):
        b = codebook.transform(feature_vector)
        # count, how often pattern appears
        b[np.nonzero(b)] = 1
        histogram = b.sum(axis=0)  
        norm_hist = normalize(histogram.reshape(1,-1))[0]
        return norm_hist
            
    def _deserialize(self, serialized):
        #self.codebook = pickle.loads(serialized[0])
        components = serialized
        # use coder instead of dictionary, performance wise
        coder = SparseCoder(components)  
        return coder
        
    def _serialize(self, codebook):        
        #serialized = pickle.dumps(self.codebook) 
        #return np.array([serialized])
        return codebook.components_  
            
    
class KMeansCodebook(Codebook):
    n_components = 50
    
    def _fit(self, feature_values):        
        # reshape to two dimensional vector (list of 1d features)
        feature_vector = feature_values.reshape(feature_values.shape[0], -1)
        pca = PCA(n_components=self.n_components).fit(feature_vector)
        codebook = KMeans(init=pca.components_, 
                          n_clusters=self.n_components,
                          n_init=1).fit(feature_vector)
        return codebook
        
    def _deserialize(self, serialized):
        codebook = pickle.loads(serialized[0])
        return codebook
    
    def _serialize(self, codebook):
        serialized = pickle.dumps(codebook)
        return np.array([serialized])
    
    def _histogram(self, codebook, feature_vector):
        prediction = codebook.predict(feature_vector)
        # which patterns appear how often?
        patterns, pattern_count = np.unique(prediction, return_counts=True)        
        histogram = np.zeros(50)
        histogram[patterns] = pattern_count
        norm_hist = normalize(histogram.reshape(1,-1))[0]
        return norm_hist
    
        
        