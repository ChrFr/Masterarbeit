from masterarbeit.model.backend.hdf5_data import HDF5Pandas
from masterarbeit.model.features.idsc import IDSCDict, IDSCKMeans, IDSCGaussiansKMeans
from masterarbeit.model.classifiers.mlp import ComplexMLP
from masterarbeit.model.classifiers.svm import SVM
from masterarbeit.model.features.moments import ZernikeMoments, HuMoments
from masterarbeit.model.segmentation.segmentation_opencv import Binarize
from masterarbeit.model.segmentation.common import read_image
from masterarbeit.model.classifiers.metrics import ConfusionMatrix
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np

def test_mlp(features, classifier):
    classifier.train(features) 
    #h5.save_classifier(mlp)
    #mlp_loaded = h5.get_classifier(MLP, 'IDSCDictTESTbatch')
    #idx = np.random.choice(len(features), 15)
    #feature_test = np.array(features)[idx]
    predicted = classifier.predict(features)    
    real = [f.category for f in features]
    print('{}/{}'.format((real==predicted).sum(), len(real)))
    
    #features = h5.get_features(ZernikeMoments, ['Klarapfel', 'roter Boskop', 'Pfirsich', 'Platane', 'Stieleiche'])  
    #mlp.train(features) 
    #predicted = mlp.predict(features)    
    #real = [f.category for f in features]
    #print('{}/{}'.format((real==predicted).sum(), len(real)))    
    
    #features = h5.get_features(HuMoments, ['Klarapfel', 'roter Boskop', 'Pfirsich', 'Platane', 'Stieleiche'])  
    #mlp.train(features) 
    #predicted = mlp.predict(features)    
    #real = [f.category for f in features]
    #print('{}/{}'.format((real==predicted).sum(), len(real))) 
    
    #from os import listdir
    #from os.path import isfile, join
    #paths = [('Platane', 'C:/Users/chris/Desktop/test/Platane')]#,
             ##('Stieleiche', 'C:/Users/chris/Desktop/test/Stieleiche')]
    #features_2 = []
    #codebook = h5.get_codebook(IDSCDict)
    #for path in paths:
        #files = [join(path[1], f) for f in listdir(path[1]) if isfile(join(path[1], f))]
        #for i, f in enumerate(files): 
            #print(f)
            #img = read_image(f)
            #binary = Binarize().process(img)
            #feature = IDSCDict(path[0])
            #feature.describe(binary)        
            #feature.histogram(codebook)
            #features_2.append(feature)
            #if i == 6: break
    #predicted = mlp_loaded.predict(features_2)
        
        
    
    #id = np.random.choice(len(values), 15)
    #np.random.shuffle(x)
    #mlp.predict(features)
    
def validation_test(features, classifier):
    #categories, values, feat_types = zip(*[(f.category, f.values, type(f)) 
                                           #for f in features])     
    #classes = np.unique(categories)
    #X_train, X_test, y_train, y_test = train_test_split(values, categories, random_state=0)
    
    #classifier._train(X_train, y_train)
    #y_pred = classifier._predict(np.array(X_test))
    real, predicted, accuracy = classifier.validate(features)
    print(accuracy)
    conf = ConfusionMatrix(real, predicted)
    conf.normalize()
    conf.plot(title='Confusion matrix normalized')
    
if __name__ == '__main__':
    h5 = HDF5Pandas()
    h5.open('../../hdf5_test.h5')     
    #features = h5.get_features(IDSCDict)      
    #features = h5.get_features(ZernikeMoments)  
    #features = h5.get_features(IDSCKMeans) 
    features = h5.get_features(IDSCGaussiansKMeans) 

    classifier = ComplexMLP('IDSCDictTESTbatch') 
    #classifier = SVM('IDSCDictTESTbatch')
    
    
    #test_mlp(features, mlp)
    validation_test(features, classifier)

    h5.close()   