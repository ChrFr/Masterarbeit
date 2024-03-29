import pytest
import numpy as np
import os

from masterarbeit.model.classifiers.mlp import ComplexMLP
from masterarbeit.model.classifiers.svm import SVM
from masterarbeit.model.features.feature import Feature
from masterarbeit.model.backend.hdf5_data import HDF5Pandas

_classifiers = [ComplexMLP, SVM]
_feature_dim = 3
_n_categories = 3
_n_feat_per_cat = 5
_data = {HDF5Pandas: 'hdf5_test.h5'}

@pytest.fixture(scope='module', params=_data)
def data(request):    
    Data = request.param
    data = Data()
    source = _data[Data]
    if os.path.exists(source):
        os.remove(source)
    data.open(source)    
    yield data
    data.close()    

# sets up all classifiers once, 
# if passed to a test, the test will be run for all of them
@pytest.fixture(scope='module', params=_classifiers)
def classifier(request):
    Classifier = request.param
    cl = Classifier('test')
    return cl

@pytest.fixture(scope='module')
def features(request):
    # mocked feature, no extraction tested here
    class TestFeature(Feature):
        columns = np.arange(_feature_dim) 
        def _describe():
            pass
    features = []    
    for n in range(_n_categories):
        cat = 'test{}'.format(n)
        for i in range(_n_feat_per_cat):
            feature = TestFeature(cat)
            feature._v = np.random.rand(_feature_dim)
            features.append(feature)
    return features

@pytest.mark.order1
def test_training(classifier, features):
    classifier.cross_validation(features)
    classifier.train(features)

@pytest.mark.order2
def test_saving(classifier, data):
    # ToDo: not valid this way atm
    pass#data.save_classifier(classifier)

@pytest.mark.order3
def test_loading(classifier, data):
    data.get_classifier(type(classifier), 'test')

if __name__ == '__main__':
    pytest.main([__file__, '-v'])