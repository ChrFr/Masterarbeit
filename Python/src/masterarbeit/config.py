import json
import os

from masterarbeit.model.preprocessor import preprocessor_opencv as pocv
from masterarbeit.model.preprocessor import preprocessor_skimage as psk
from masterarbeit.model.features.hu_moments import HuMoments
from masterarbeit.model.features.zernike_moments import ZernikeMoments
from masterarbeit.model.backend.hdf5_data import HDF5Pandas

IMAGE_FILTER = 'Images (*.png, *.jpg)'
ALL_FILES_FILTER = 'All Files(*.*)'
HDF5_FILTER = 'HDF5 (*.h5)'

PRE_PROCESSORS = (pocv.Binarize, psk.BinarizeHSV, psk.SegmentGabor, 
                  pocv.SegmentVeinsGabor)
FEATURES = (HuMoments, ZernikeMoments)
DATA = [HDF5Pandas]


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
    
class Config(metaclass=Singleton):  
    
    config_file = 'config.json'
    
    _default = {
        'data': HDF5Pandas,
        'source': os.path.join(os.getcwd(), 'default_store.h5')
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
            json.dump(config_copy, f)
    
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
        