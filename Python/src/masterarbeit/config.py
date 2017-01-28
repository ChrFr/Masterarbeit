import json
import os

from masterarbeit.model.preprocessor import preprocessor_opencv as pocv
from masterarbeit.model.preprocessor import preprocessor_skimage as psk
from masterarbeit.model.features.hu_moments import HuMoments
from masterarbeit.model.features.zernike_moments import ZernikeMoments
from masterarbeit.model.backend.hdf5_data import HDF5Pandas


IMAGE_FILTER = 'Images (*.png, *.jpg)'
ALL_FILES_FILTER = 'All Files(*.*)'

PRE_PROCESSORS = (pocv.Binarize, psk.BinarizeHSV, psk.SegmentGabor, 
                  pocv.SegmentVeinsGabor)
FEATURES = (HuMoments, ZernikeMoments)
DATA = (HDF5Pandas)

class Config():  
    config_file = 'config.json'
    
    _config_dict = {
        'data': HDF5Pandas.__name__,
        'source': 'default_store.h5'
        }
    
    def __init__(self):
        if os.path.exists(self.config_file):            
            self.read()
        else:
            self.write() # write default config, if file doesn't exist yet
        
    def read(self):
        try:
            with open(self.config_file, 'r') as cf:
                self._config_dict = json.load(cf)
        except:
            print('Error while loading config. Using default values.')
    
    def write(self):    
        with open(self.config_file, 'w') as f:
            json.dump(self._config_dict, f)
        
        