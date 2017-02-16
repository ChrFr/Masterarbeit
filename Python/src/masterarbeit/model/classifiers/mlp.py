import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import copy
import pickle

from masterarbeit.model.classifiers.classifier import Classifier
from masterarbeit.model.features.common import features_to_dataframe
from masterarbeit.model.backend.data import load_class, class_to_string

class MLP(Classifier):   
    label = 'Multilayer Perceptron'
    verbose = 2
    epoch = 300
    batch_size = 16
    
    def __init__(self, name):
        super(MLP, self).__init__(name)
        self.meta = {}
        self.model = Sequential()
            
    def _train(self, values, classes, n_classes):
        encoded_categories = np_utils.to_categorical(classes)
        input_dim = values.shape[1]
        
        self.setup_model(input_dim, n_classes)        
        self.model.fit(values, encoded_categories, 
                       nb_epoch=self.epoch, batch_size=self.batch_size, 
                       verbose=self.verbose)
        
    def _predict(self, values):
        predictions = self.model.predict(values, batch_size=16)   
        classes = np_utils.probas_to_classes(predictions)
        return classes
    
    def serialize(self):
        trained_features = [class_to_string(ft) for ft in self.trained_features]
        return [trained_features, self.trained_categories]
    
    def deserialize(self, serialized):
        self.trained_features = [load_class(ft) for ft in serialized[0]]
        self.trained_categories = serialized[1]
        
        
class ComplexMLP(MLP):
    label = 'Multilayer Perceptron 64x64'
    dense_layers = 1
    activation = 'relu'
    hidden_units = 64
    def setup_model(self, input_dim, n_categories):    
        self.input_dim = input_dim
        for layer in range(self.dense_layers):
            self.model.add(Dense(self.hidden_units,
                                 input_dim=input_dim, init='uniform'))
            self.model.add(Activation(self.activation))
            self.model.add(Dropout(0.5))
        self.model.add(Dense(n_categories, init='uniform'))
        self.model.add(Activation('softmax'))
    
        #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])         
        
        
class SimpleMLP(MLP):
    label = 'Multilayer Perceptron 4'
    def setup_model(self, input_dim, n_categories):
        self.model.add(Dense(4, input_dim=input_dim, init='normal', activation='relu'))
        self.model.add(Dense(n_categories, init='normal', activation='sigmoid'))
        # Compile model
        self.model.compile(loss='categorical_crossentropy', 
                           optimizer='adam', metrics=['accuracy'])     
  
    