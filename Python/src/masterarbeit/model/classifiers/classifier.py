'''
contains the abstract class of a classifier

(c) 2017, Christoph Franke

this file is part of the master thesis 
"Computergestuetzte Identifikation von Pflanzen anhand ihrer Blattmerkmale"
'''
__author__ = "Christoph Franke"

import numpy as np
from abc import ABCMeta
from abc import abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import (label_ranking_loss, 
                             label_ranking_average_precision_score, 
                             coverage_error, average_precision_score)

from keras.utils import np_utils
from masterarbeit.model.features.feature import UnorderedFeature
from masterarbeit.model.backend.data import load_class, class_to_string

class Classifier(metaclass=ABCMeta):       
    label = 'None'
    validation_split = 0.33
    
    def __init__(self, name, seed=None):
        self.seed = seed
        self.name = name
        self.model = None
        self.trained_features = []
        self.trained_categories = []
        self.input_dim = 0
        
    @abstractmethod
    def setup_model(self, input_dim, n_categories):
        pass
          
    def train(self, features):
        
        categories, values, feat_types, codebook_types = zip(
            *[(f.category, f.values, type(f), 
               f.codebook_type if hasattr(f, 'codebook_type') else None) 
              for f in features]) 
        feat_types = np.array(feat_types)
        codebook_types = np.array(codebook_types)
        u, idx = np.unique(feat_types.astype(str), 
                           return_index=True)
        unique_types = feat_types[idx]
        
        codebooks_used = codebook_types[idx]
        self.trained_features = dict(zip(unique_types, codebook_types))
        
        # returned index serves as an integer representation of the category 
        # with SAME order of the unique categories (important for 
        # reversing class to category in prediction)
        unique_cat, classes, count = np.unique(categories, 
                                               return_inverse=True, 
                                               return_counts=True)
        self.trained_categories = list(unique_cat)
        n_classes = len(unique_cat)
        values = np.array(values)
        values = np.nan_to_num(values)
        print('Start training for {} categories '.format(n_classes) +
              'with an input dimension of {}'.format(values.shape[1]))
        self._train(values, classes, n_classes)
            
    @abstractmethod
    def _train(self, values, classes):
        '''
        values numpy array
        '''
        pass
    
    def predict(self, features):    
        #if feat_type not in self.trained_feature_types
        values = np.array([feat.values for feat in features])   
        values = np.nan_to_num(values)
        predictions = self._predict(values)        
        classes = np_utils.probas_to_classes(predictions)        
        return np.array(np.array(self.trained_categories))[classes]
    
    def _features_to_values(self, features):
        categories, values = zip(*[(f.category, f.values) for f in features])
        unique_cat, classes = np.unique(categories, return_inverse=True)
        return np.array(values), classes, unique_cat
            
    def cross_validation(self, features):
        '''
        standalone validation of an untrained classifier
        splits the features into a training test set and a set for validation
        Warning: overwrites existing trained model 
        '''
        values, classes, categories = self._features_to_values(features)
        values = np.nan_to_num(values)
        n_classes = len(categories)
        
        (training_values, test_values, 
         training_classes, test_classes) = train_test_split(
             values, classes, test_size=self.validation_split, 
             random_state=self.seed)
        
        self._train(np.array(training_values), training_classes, n_classes)        
        predictions = self._predict(np.array(test_values))           
        predicted_classes = np_utils.probas_to_classes(predictions)    
        binary_labels = np_utils.to_categorical(test_classes)
        
        # compute the metrics
        accuracy = accuracy_score(test_classes, predicted_classes)
        precision_score = average_precision_score(binary_labels, predictions)
        error = coverage_error(binary_labels, predictions)
        loss = label_ranking_loss(binary_labels, predictions)
        label_precision = label_ranking_average_precision_score(binary_labels, 
                                                          predictions)
        
        real_cat = categories[test_classes]
        predicted_cat = categories[predicted_classes]         
        return (real_cat, predicted_cat, accuracy, precision_score, 
                error, loss, label_precision)
    
    def serialize(self):
        trained_features = [class_to_string(ft)
                            for ft in self.trained_features.keys()]
        codebooks_used = [class_to_string(c) 
                          for c in self.trained_features.values()]
        return [trained_features, codebooks_used, self.trained_categories]
    
    def deserialize(self, serialized):
        trained_features = [load_class(ft) for ft in serialized[0]]
        codebooks_used = [load_class(c) for c in serialized[1]]
        self.trained_features = dict(zip(trained_features, codebooks_used))
        self.trained_categories = serialized[2]     
            
    @abstractmethod        
    def _predict(self, values):
        pass        

if __name__ == '__main__':    
    pass