from masterarbeit.model.backend.data import Data
import pandas as pd
import numpy as np
import datetime
import os

class HDF5Data(Data):
    def __init__(self):
        self.store = None
    
    def open(self, source):
        self.store = pd.get_store(source)
        
    def get_species(self):   
        # the upper folders represent the names of the species
        names = [os.path.split(k)[0].strip('/') for k in self.store.keys()]
        return names
    
    def add_feature(self, species, feature):
        path = '{s}/{f}'.format(s=species,f=feature.name)        
        now = datetime.datetime.now().strftime("%H:%M:%S-%d.%m.%y")
        df = pd.DataFrame([feature.values], columns=np.arange(len(feature.values)), index=[now])      
        self.store.append(path, df)
        
    def close(self):        
        self.store.close()