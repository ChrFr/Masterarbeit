'''
contains implementations of Multilayer Perceptrons

(c) 2017, Christoph Franke

this file is part of the master thesis 
"Computergestuetzte Identifikation von Pflanzen anhand ihrer Blattmerkmale"
'''
__author__ = "Christoph Franke"

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import copy
import pickle
from sklearn.model_selection import train_test_split
THEANO_FLAGS='floatX=float32,device=cuda0,lib.cnmem=1'

from masterarbeit.model.classifiers.classifier import Classifier

class MLP(Classifier):   
    label = 'Multilayer Perceptron'
    verbose = 2
    epoch = 300
    batch_size = 16
    
    def __init__(self, name, seed=None):
        super(MLP, self).__init__(name, seed=seed)
        self.meta = {}
        self.model = Sequential()
            
    def _train(self, values, classes, n_classes):
        binary_labels = np_utils.to_categorical(classes)
        input_dim = values.shape[1]
        
        self.setup_model(input_dim, n_classes)        
        self.model.fit(values, binary_labels, 
                       nb_epoch=self.epoch, 
                       batch_size=self.batch_size, 
                       verbose=self.verbose)
        
    def validation_history(self, features):
        '''
        Keras exclusive, used to evaluate overfitting
        ToDo: integrate in validation of Classifier class
        '''
        values, classes, categories = self._features_to_values(features)        
        input_dim = values.shape[1]
        n_classes = len(categories)
        
        # manual split, keras makes strange things else        
        (training_values, test_values, 
         training_classes, test_classes) = train_test_split(
             values, classes, test_size=self.validation_split, 
             random_state=self.seed)
    
        self.setup_model(training_values.shape[1], n_classes)      
        training_classes = np_utils.to_categorical(training_classes)
        test_classes = np_utils.to_categorical(test_classes)
        history = self.model.fit(training_values, training_classes, 
                                 validation_data=(test_values, test_classes), 
                                 nb_epoch=self.epoch, 
                                 batch_size=self.batch_size,
                                 verbose=self.verbose)      
        return history
                
    def _predict(self, values):
        predictions = self.model.predict(values, batch_size=16)   
        return predictions
    
    
class SimpleMLP(MLP):
    label = 'Simple Multilayer Perceptron'
    def setup_model(self, input_dim, n_categories):
        self.model.add(Dense(10, input_dim=input_dim, init='normal', 
                             activation='relu'))
        self.model.add(Dense(n_categories, init='normal', activation='sigmoid'))
        # Compile model
        self.model.compile(loss='categorical_crossentropy', 
                           optimizer='adam', 
                           metrics=['accuracy'])                
        
        
class ComplexMLP(MLP):
    label = 'Complex adjustable Multilayer Perceptron'
    activation = 'tanh'
    hidden_units = [256]
    initial_dropout = True
    dropout = 0.2
    
    def setup_model(self, input_dim, n_categories):    
        self.input_dim = input_dim
        # dropout layer between visible and first hidden layer
        if self.initial_dropout and self.dropout > 0:
            self.model.add(Dropout(self.dropout, input_shape=(input_dim,)))
            # following layers don't need this information if defined once
            input_dim = None
        # hidden layers
        for hidden in self.hidden_units:
            self.model.add(Dense(hidden, init='uniform', input_dim=input_dim))
            self.model.add(Activation(self.activation))
            # dropout layer between this and next layer
            if self.dropout > 0:
                self.model.add(Dropout(self.dropout))
        self.model.add(Dense(n_categories, init='uniform'))
        self.model.add(Activation('sigmoid'))       
    
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam', # sgd testet, but worse performance
                           metrics=['accuracy'])                 
        
  
    