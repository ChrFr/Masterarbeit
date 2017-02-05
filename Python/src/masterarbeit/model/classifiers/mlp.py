from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
import copy

from masterarbeit.model.classifiers.classifier import Classifier
from masterarbeit.model.features.common import features_to_dataframe

class MLP(Classifier):   
    label = 'Multilayer Perceptron'
    
    def __init__(self, name):
        super(MLP, self).__init__(name)
        self.model = Sequential()
    
    def setup_model(self, input_dim):
        print('setting up model')
        self.model.add(Dense(4, input_dim=input_dim, init='normal', activation='relu'))
        self.model.add(Dense(3, init='normal', activation='sigmoid'))
        # Compile model
        self.model.compile(loss='categorical_crossentropy', 
                           optimizer='adam', metrics=['accuracy'])  
        return self.model
            
    def train(self, features):
        category, values = zip(*[(f.category, f.values) for f in features])       
        
        encoder = LabelEncoder()
        encoder.fit(category)
        encoded_category = encoder.transform(category)
        
        unique_cat, categories = np.unique(encoded_category, return_inverse=True)
        # convert integers to dummy variables (i.e. one hot encoded)
        categories = np_utils.to_categorical(categories)
        self.model.fit(np.array(values), categories)
        
    def validate(self, features):
        category, values = zip(*[(f.category, f.values) for f in features])       
        
        encoder = LabelEncoder()
        encoder.fit(category)
        encoded_category = encoder.transform(category)
        
        unique_cat, categories = np.unique(encoded_category, return_inverse=True)
        categories = np_utils.to_categorical(categories)
        seed = 7
        input_dim = len(features[0])     
        
        def baseline_model(input_dim):
            # create model
            model = Sequential()
            model.add(Dense(4, input_dim=input_dim, init='normal', activation='relu'))
            model.add(Dense(3, init='normal', activation='sigmoid'))
            # Compile model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model
        
        estimator = KerasClassifier(build_fn=lambda: self.setup_model(input_dim),
                                    nb_epoch=200, batch_size=5, verbose=0)
        
        def get_params_fix(classifier, deep=False):
            res = copy.deepcopy(classifier.sk_params)
            res.update({'build_fn': classifier.build_fn})
            return res 
        
        KerasClassifier.get_params = get_params_fix    
    
        kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
        results = cross_val_score(estimator, np.array(values), categories, cv=kfold)
        print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))          
    
    def load(self, source):
        pass
    
    def save(self, source):
        pass
    
    def predict(self, features):
        pass    