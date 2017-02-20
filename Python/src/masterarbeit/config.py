import json
import os

from masterarbeit.model.segmentation import segmentation_opencv as pocv
from masterarbeit.model.segmentation import segmentation_skimage as psk
from masterarbeit.model.features.moments import ZernikeMoments, HuMoments
from masterarbeit.model.features.texture import Sift, SiftPatch, LocalBinaryPattern, LocalBinaryPatternCenterPatch, Leafvenation, GaborFilterBank, GaborFilterBankPatches, LocalBinaryPatternPatches, GaborFilterBankCenterPatch, Haralick, Surf, SurfPatch, LocalBinaryPatternKMeans
from masterarbeit.model.features.borders import Borders
from masterarbeit.model.features.idsc import (IDSCKMeans, IDSCDict, 
                                              IDSCGaussiansKMeans,
                                              IDSCGaussiansDict)
from masterarbeit.model.backend.hdf5_data import HDF5Pandas
from masterarbeit.model.classifiers.mlp import ComplexMLP, SimpleMLP
from masterarbeit.model.classifiers.svm import SVM

IMAGE_FILTER = 'Images (*.png, *.jpg)'
ALL_FILES_FILTER = 'All Files(*.*)'
HDF5_FILTER = 'HDF5 (*.h5)'

SEGMENTATION = (pocv.Binarize, psk.BinarizeHSV, 
                pocv.KMeansBinarize, pocv.KMeansHSVBinarize)
FEATURES = (HuMoments, ZernikeMoments, Borders, IDSCKMeans, IDSCDict,
            IDSCGaussiansKMeans, IDSCGaussiansDict, Sift, SiftPatch, LocalBinaryPattern, LocalBinaryPatternCenterPatch, Leafvenation, GaborFilterBank, GaborFilterBankPatches, LocalBinaryPatternPatches, GaborFilterBankCenterPatch, Haralick, Surf, SurfPatch, LocalBinaryPatternKMeans)
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
    default_segmentation = pocv.KMeansHSVBinarize
    
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
        