'''
contains the configuration for the UI

(c) 2017, Christoph Franke

this file is part of the master thesis 
"Computergestuetzte Identifikation von Pflanzen anhand ihrer Blattmerkmale"
'''
__author__ = "Christoph Franke"

import json
import os

from masterarbeit.model.segmentation import segmentation
from masterarbeit.model.features import moments
from masterarbeit.model.features import texture
from masterarbeit.model.features import keypoint_features
from masterarbeit.model.features import idsc
from masterarbeit.model.backend.hdf5_data import HDF5Pandas
from masterarbeit.model.classifiers import mlp
from masterarbeit.model.classifiers import svm
from masterarbeit.model.features import codebooks

SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.tif']
IMAGE_FILTER = 'Images ({})'.format(
    ','.join(['*' + s for s in SUPPORTED_IMAGE_EXTENSIONS]))
ALL_FILES_FILTER = 'All Files(*.*)'
HDF5_FILTER = 'HDF5 (*.h5)'

SEGMENTATION = (
    segmentation.Binarize, 
    segmentation.BinarizeHSV, 
    segmentation.KMeansBinarize, 
    segmentation.KMeansHSVBinarize, 
    segmentation.Slic
)

FEATURES = (
    moments.HuMoments, moments.ZernikeMoments,
    idsc.MultilevelIDSC, idsc.IDSC, 
    keypoint_features.Sift, keypoint_features.Surf, 
    texture.LocalBinaryPattern, 
    texture.LocalBinaryPatternCenterPatch, 
    texture.LocalBinaryPatternPatches,
    texture.LeafvenationMorph, 
    texture.GaborFilterBank, 
    texture.GaborFilterBankPatches, 
    texture.GaborFilterBankCenterPatch)

CODEBOOKS = (
    codebooks.KMeansCodebook, 
    codebooks.DictLearningCodebook
)

CLASSIFIERS = (
    mlp.SimpleMLP, 
    mlp.ComplexMLP, 
    svm.SVM
)

DATA = [HDF5Pandas]

file_path = os.path.split(__file__)[0]


class Singleton(type):
    """Metaclass for singletons    
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
    
class Config(metaclass=Singleton):  
    """Class holding the configuration for the UI.
    
    configurations are accessed dict-like
    
    Attributes:
        config_file (str): the filename of the configuration file

    """
    # config file is stored in same path as this file
    config_file = os.path.join(file_path, 'config.json')
    
    # the main segmentation used, when not asked for specific one
    default_segmentation = segmentation.KMeansHSVBinarize
    
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
        """read the configuration file
        
        reads configuration file and sets stored configurations       
        
        """
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
        """write to the configuration file
        
        writes configuration to the configuration file as JSON
        
        """
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
        