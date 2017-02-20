import numpy as np
seed = 0
np.random.seed = seed
PYTHONHASHSEED = seed
from masterarbeit.model.backend.hdf5_data import HDF5Pandas
from masterarbeit.model.features.idsc import IDSCDict, IDSCKMeans, IDSCGaussiansKMeans, IDSCGaussiansDict
from masterarbeit.model.features.texture import LocalBinaryPattern, LocalBinaryPatternCenterPatch, GaborFilterBank, GaborFilterBankPatches, LocalBinaryPatternPatches, GaborFilterBankCenterPatch, Haralick, Leafvenation, Sift, SiftPatch, Surf, SurfPatch
from masterarbeit.model.classifiers.mlp import ComplexMLP
from masterarbeit.model.classifiers.svm import SVM
from masterarbeit.model.features.moments import ZernikeMoments, HuMoments
from masterarbeit.model.segmentation.segmentation_opencv import Binarize
from masterarbeit.model.segmentation.helpers import read_image
from masterarbeit.model.classifiers.metrics import ConfusionMatrix
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

    
def validation_test(features, classifier):
    #categories, values, feat_types = zip(*[(f.category, f.values, type(f)) 
                                           #for f in features])     
    #classes = np.unique(categories)
    #X_train, X_test, y_train, y_test = train_test_split(values, categories, random_state=0)
    
    #classifier._train(X_train, y_train)
    #y_pred = classifier._predict(np.array(X_test))
    real, predicted, accuracy = classifier.validate(features)
    print(accuracy)
    #conf = ConfusionMatrix(real, predicted)
    #conf.normalize()
    #conf.plot(title='Confusion matrix normalized')
    
def test_feat_types(feat_types, species, h5):
    for feat_type in feat_types:
        features = h5.get_features(feat_type, species) 
        
        classifier = ComplexMLP('IDSCDictTESTbatch') 
        validation_test(features, classifier)    
        classifier = SVM('IDSCDictTESTbatch', seed=0)
        validation_test(features, classifier)    
        #test_mlp(features, mlp)    
        
def test_join(feat_types, species, h5):
    joined = h5.get_joined_features(feat_types, species)
    classifier = ComplexMLP('IDSCDictTESTbatch') 
    validation_test(joined, classifier)     
    classifier = SVM('IDSCDictTESTbatch', seed=0)
    validation_test(joined, classifier)
    
def test_mlp_def(feat_type, h5):
    features = h5.get_features(feat_type, species) 
    
    for hidden_units in [32, 64, 128, 256]:
        print('hidden units: {}'.format(hidden_units))
        classifier = ComplexMLP('IDSCDictTESTbatch')
        classifier.hidden_units = hidden_units
        validation_test(features, classifier) 
        
    for dense_layers in range(1, 4):
        print('dense layers: {}'.format(dense_layers))
        classifier = ComplexMLP('IDSCDictTESTbatch')
        classifier.dense_layers = dense_layers
        validation_test(features, classifier) 
        
    for activation in ['sigmoid', 'hard_sigmoid', 'relu', 'tanh', 'softplus', 'softsign', 'linear']:
        print(activation)
        classifier = ComplexMLP('IDSCDictTESTbatch')
        classifier.activation = activation
        validation_test(features, classifier)         
    
if __name__ == '__main__':
    h5 = HDF5Pandas()
    #h5.open('../../hdf5_test.h5') 
    #h5.open('../../contour_test.h5')  
    h5.open('../../eigenes_set.h5')    
    species = None#['1 - Klarapfel', '2 - roter Boskop', '7 - Apfelquitte']
    feat_types = [IDSCGaussiansKMeans, LocalBinaryPattern, LocalBinaryPatternCenterPatch, LocalBinaryPatternPatches] #IDSCGaussiansKMeans, IDSCDict, ZernikeMoments, IDSCKMeans, LocalBinaryPatternPatch
    
    ComplexMLP.verbose = 0    
    ComplexMLP.epoch = 2000
    ComplexMLP.activation = 'softsign'
    #test_feat_types([IDSCDict], species, h5)
    #test_feat_types([IDSCDict], species, h5)
    #test_feat_types([IDSCKMeans], species, h5)
    #print('-' * 10)  
    #test_feat_types([IDSCGaussiansDict], species, h5)
    #test_feat_types([IDSCGaussiansDict], species, h5)
    #test_feat_types([IDSCGaussiansDict], species, h5)
    print('-' * 10) 
    #test_feat_types([IDSCGaussiansKMeans], species, h5)
    #test_feat_types([IDSCGaussiansKMeans], species, h5)
    #test_feat_types([IDSCGaussiansKMeans], species, h5)
    
    #test_mlp_def(IDSCGaussiansKMeans, h5)
    
    #test_feat_types([GaborFilterBankCenterPatch], species, h5)   
    #test_feat_types([GaborFilterBankCenterPatch], species, h5)   
    #test_mlp_def(feat_types[0], species, h5)
    print('joined')
    test_join([IDSCGaussiansDict, Surf], species, h5)
    test_join([IDSCGaussiansDict, GaborFilterBankPatches], species, h5)
    test_join([IDSCGaussiansDict, Surf, GaborFilterBankPatches], species, h5)
    test_join([IDSCGaussiansKMeans, Surf, GaborFilterBankPatches], species, h5)
    #test_join([IDSCGaussiansKMeans, LocalBinaryPattern], species, h5)
    #test_join([IDSCGaussiansKMeans, LocalBinaryPattern], species, h5)
    #test_join([IDSCGaussiansKMeans, LocalBinaryPatternCenterPatch], species, h5)
    #test_join([IDSCGaussiansKMeans, LocalBinaryPatternCenterPatch], species, h5)
    #test_join([IDSCGaussiansKMeans, LocalBinaryPatternCenterPatch], species, h5)
    
    #print('-' * 10)  
    #test_feat_types([IDSCGaussiansKMeans], species, h5)  
    #print('-' * 10)  
    #test_feat_types([Surf], species, h5)  
    #print('-' * 10)
    #test_join([IDSCGaussiansKMeans, GaborFilterBankPatches], species, h5)
    #test_join([IDSCGaussiansKMeans, GaborFilterBankPatches], species, h5)
    #test_join([IDSCGaussiansKMeans, GaborFilterBankPatches], species, h5)
    #print('-' * 10)
    #test_join([IDSCGaussiansKMeans, Surf, GaborFilterBankPatches], species, h5)
    #test_join([IDSCGaussiansKMeans, Surf, GaborFilterBankPatches], species, h5)
    #test_join([IDSCGaussiansKMeans, Surf, GaborFilterBankPatches], species, h5)
    #print('-' * 10)
    #test_join([IDSCGaussiansKMeans, Surf, Sift, GaborFilterBankPatches], species, h5)
    #test_join([IDSCGaussiansKMeans, Surf, Sift, GaborFilterBankPatches], species, h5)
    #test_join([IDSCGaussiansKMeans, Surf, Sift, GaborFilterBankPatches], species, h5)
    #test_feat_types([Surf], species, h5)  
    #print('-' * 10)
    #test_feat_types([Sift], species, h5)  
    #print('-' * 10)
    #test_feat_types([IDSCGaussiansKMeans], species, h5)  
    #print('-' * 10)
    #test_join([IDSCGaussiansKMeans, Surf], species, h5)
    #print('-' * 10)
    #test_join([IDSCGaussiansKMeans, Sift], species, h5)
    #print('-' * 10)
    #test_join([IDSCGaussiansKMeans, GaborFilterBankPatches], species, h5)
    #print('-' * 10)
    #test_join([GaborFilterBankPatches, Surf], species, h5)
    #print('-' * 10)
    #test_join([GaborFilterBankPatches, Sift], species, h5)
    #print('-' * 10)
    #test_join([Leafvenation, LocalBinaryPatternPatches, GaborFilterBankPatches], species, h5)
    #print('-' * 10)
    #test_join([IDSCGaussiansKMeans, Leafvenation, LocalBinaryPatternPatches], species, h5)
    #print('-' * 10)
    #test_join([IDSCGaussiansKMeans, Leafvenation, LocalBinaryPatternPatches, GaborFilterBankPatches], species, h5)
    #test_feat_types([Leafvenation], species, h5)   
    #test_feat_types([LocalBinaryPatternCenterPatch], species, h5)   
    #test_feat_types([LocalBinaryPatternPatches], species, h5)  
    #print('-' * 10)
    #test_feat_types([Sift], species, h5)
    #test_feat_types([Sift], species, h5)
    #print('-' * 10) 
    #test_feat_types([SiftPatch], species, h5)    
    #print('-' * 10)
    #test_feat_types([Surf], species, h5)     
    #print('-' * 10)
    #test_feat_types([SurfPatch], species, h5)  
    
    #print('-' * 10)
    #test_join([Surf, LocalBinaryPatternPatches], species, h5)
    #print('-' * 10)
    #test_join([IDSCGaussiansKMeans, Leafvenation], species, h5)    
    #print('-' * 10)
    #test_join([IDSCGaussiansKMeans, GaborFilterBankPatches], species, h5)
    #print('-' * 10)
    #test_join([IDSCGaussiansKMeans, GaborFilterBankPatches], species, h5)

    #print('-' * 10)
    #test_join([Surf, Leafvenation], species, h5)
    #print('-' * 10)
    #test_join([Sift, Leafvenation], species, h5)
    
    #test_join([IDSCGaussiansKMeans, GaborFilterBank], species, h5)
    #test_join([IDSCGaussiansKMeans, GaborFilterBank], species, h5)
    #test_join(feat_types, species, h5)
    #print('gauss')
    #test_feat_types([feat_types[1]], species, h5)
    #test_feat_types([feat_types[1]], species, h5)
    #test_feat_types([feat_types[1]], species, h5)
    
    h5.close()   