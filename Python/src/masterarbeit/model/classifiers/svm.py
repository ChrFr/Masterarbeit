'''
contains an implementation of a Support Vector Machine

(c) 2017, Christoph Franke

this file is part of the master thesis 
"Computergestuetzte Identifikation von Pflanzen anhand ihrer Blattmerkmale"
'''
__author__ = "Christoph Franke"

from masterarbeit.model.classifiers.classifier import Classifier
from masterarbeit.model.backend.data import load_class, class_to_string
from sklearn.svm import SVC
import numpy as np
import pickle

class SVM(Classifier):   
    
    label = 'Support Vector Machine'
    
    def __init__(self, name, seed=None):
        super(SVM, self).__init__(name, seed=seed)
        self.meta = {}
        
    def setup_model(self, input_dim, n_classes):    
        self.input_dim = input_dim
        # if random_state is None, no predefined seed will be used by SVM
        self.model = SVC(kernel='linear', probability=True, 
                         random_state=self.seed,
                         decision_function_shape='ovr') 

    def _train(self, values, classes, n_classes):
        input_dim = values.shape[1]
    
        self.setup_model(input_dim, n_classes)   
        values = np.nan_to_num(values)
        self.model.fit(values, classes)

    def _predict(self, values):
        values = np.nan_to_num(values)
        predictions = self.model.predict_proba(values)
        return predictions
    
    def serialize(self):
        pickled = pickle.dumps(self.model)
        trained_features = [class_to_string(ft) for ft in self.trained_features]
        return [[pickled], trained_features, self.trained_categories]
    
    def deserialize(self, serialized):
        self.model = pickle.loads(serialized[0][0])
        self.trained_features = [load_class(ft) for ft in serialized[1]]
        self.trained_categories = serialized[2] 