import json
import os

from masterarbeit.model.segmentation.segmentation import (
    Binarize, BinarizeHSV, KMeansBinarize, KMeansHSVBinarize, Slic)
from masterarbeit.model.features.moments import ZernikeMoments, HuMoments
from masterarbeit.model.features.texture import (
    Sift, Surf,
    LocalBinaryPattern, LocalBinaryPatternCenterPatch, 
    LocalBinaryPatternPatches,
    LeafvenationMorph, 
    GaborFilterBank, GaborFilterBankPatches,
    GaborFilterBankCenterPatch)
from masterarbeit.model.features.idsc import (IDSC, MultilevelIDSC)
from masterarbeit.model.backend.hdf5_data import HDF5Pandas
from masterarbeit.model.classifiers.mlp import ComplexMLP, SimpleMLP
from masterarbeit.model.classifiers.svm import SVM
from masterarbeit.model.features.codebook import (KMeansCodebook, 
                                                  DictionaryLearning)

SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.tif']
IMAGE_FILTER = 'Images ({})'.format(
    ','.join(['*' + s for s in SUPPORTED_IMAGE_EXTENSIONS]))
ALL_FILES_FILTER = 'All Files(*.*)'
HDF5_FILTER = 'HDF5 (*.h5)'

SEGMENTATION = (Binarize, BinarizeHSV, KMeansBinarize, KMeansHSVBinarize, Slic)
FEATURES = (HuMoments, ZernikeMoments,
            MultilevelIDSC, IDSC, 
            Sift, Surf, 
            LocalBinaryPattern, LocalBinaryPatternCenterPatch, 
            LocalBinaryPatternPatches,
            LeafvenationMorph, 
            GaborFilterBank, GaborFilterBankPatches, 
            GaborFilterBankCenterPatch)
CODEBOOKS = (KMeansCodebook, DictionaryLearning)
CLASSIFIERS = (ComplexMLP, SimpleMLP, SVM)
DATA = [HDF5Pandas]

file_path = os.path.split(__file__)[0]


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
    
class Config(metaclass=Singleton):  
    # config file is stored in same path as this file
    config_file = os.path.join(file_path, 'config.json')
    
    # the main segmentation used, when not asked for specific one
    default_segmentation = KMeansHSVBinarize
    
    # config is built out of this dict, if not exists yet
    _default = {
        'data': HDF5Pandas,
        'source': os.path.join(file_path, 'default_store.h5')
        }

    _config = {}
    
    def __init__(self):
        if os.path.exists(self.config_file):            
            self.read()
        # write default config, if file doesn't exist yet
        else:
            self._config = self._default.copy()
            self.write() 
        
    def read(self):
        try:
            with open(self.config_file, 'r') as cf:
                self._config = json.load(cf)
            for data in DATA:
                if self._config['data'] == data.__name__:
                    self._config['data'] = data
                    break
        except:
            self._config = self._default.copy()
            print('Error while loading config. Using default values.')
    
    def write(self):    
        # TODO: serialize class instead of storing name
        config_copy = self._config.copy()
        config_copy['data'] = self._config['data'].__name__
        
        with open(self.config_file, 'w') as f:
            # pretty print to file
            json.dump(config_copy, f, indent=4, separators=(',', ': '))
    
    # access stored config entries like fields        
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self._config:
            return self._config[name]
        raise AttributeError
    
    def __setattr__(self, name, value):   
        if name in self._config:
            self._config[name] = value  
        else:
            self.__dict__[name] = value   
        