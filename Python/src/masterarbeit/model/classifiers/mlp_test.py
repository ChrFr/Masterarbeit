from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import seaborn as sns
import copy
import numpy as np
from masterarbeit.model.features.plot import plot
from keras.models import load_model

from masterarbeit.model.backend.hdf5_data import HDF5Pandas
from masterarbeit.model.features.hu_moments import HuMoments
from masterarbeit.model.features.zernike_moments import ZernikeMoments

class MLP():
    def __init__(self, feature_class):
        self.feature_class = feature_class
        
    def train(self, dataframe):
        iris = sns.load_dataset("iris")
        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)
        # load dataset
        #dataframe = pandas.read_csv("iris.csv", header=None)
        dataset = dataframe.values
        length = dataset.shape[1] - 1
        X = dataset[:,0:length].astype(float)
        Y = dataset[:,length]
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        # convert integers to dummy variables (i.e. one hot encoded)
        uniques, ey = np.unique(encoded_Y, return_inverse=True)
        dummy_y = np_utils.to_categorical(ey)
        # define baseline model
        def baseline_model():
            # create model
            model = Sequential()
            model.add(Dense(4, input_dim=length, init='normal', activation='relu'))
            model.add(Dense(3, init='normal', activation='sigmoid'))
            # Compile model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model
        estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
        
        KerasClassifier.get_params = get_params_fix    
        
        #kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
        #results = cross_val_score(estimator, X, dummy_y, cv=kfold)
        #print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))    
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.10, random_state=np.random.randint(0, 100000))
        estimator.fit(X_train, Y_train)
        predictions = estimator.predict(X_test)       
        
        print(encoder.inverse_transform(predictions))    
        trained = uniques[Y_test.argmax(1)]
        print(encoder.inverse_transform(trained)) 
        a= (predictions==trained)
        print('{}/{} recognized correctly'.format(a.sum(), len(a)))
        print('done')
    
def get_params_fix(classifier, deep=False):
    res = copy.deepcopy(classifier.sk_params)
    res.update({'build_fn': classifier.build_fn})
    return res 


if __name__ == '__main__':
    
    h5 = HDF5Pandas()
    h5.open('../../batch_test.h5')     
    features = h5.read_feature(ZernikeMoments, ['Klarapfel', 'Platane', 'Sommerlinde'])    
    h5.close() 
    a=list(np.arange(1,10))
    a.append('species')
    #p=features[a]
    #plot(p)
    mlp = MLP(ZernikeMoments)
    mlp.train(features)    
    h5.close()