from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np

from masterarbeit.model.classifiers.classifier import Classifier

class MLP(Classifier):   
    
    def setup(self, input_dim):
        self.model = Sequential()
        self.model.add(Dense(4, input_dim=input_dim, init='normal', activation='relu'))
        self.model.add(Dense(3, init='normal', activation='sigmoid'))
        # Compile model
        self.model.compile(loss='categorical_crossentropy', 
                           optimizer='adam', metrics=['accuracy'])        
    
    def train(self, features):
        species, values = zip(*[(f.species, f.values) for f in features])       
        
        encoder = LabelEncoder()
        encoder.fit(species)
        encoded_species = encoder.transform(species)
        
        unique_cat, categories = np.unique(encoded_species, return_inverse=True)
        # convert integers to dummy variables (i.e. one hot encoded)
        categories = np_utils.to_categorical(categories)
        self.model.fit(np.array(values), categories)
    
    def load(self, source):
        pass
    
    def save(self, source):
        pass
    
    def predict(self, features):
        pass    