from masterarbeit.model.backend.data import Data
import h5py

class HDF5Data(Data):
    def __init__(self):
        self.h5f = None
    
    def open(self, source):
        self.h5f = h5py.File(source, "w")
        
    def get_species(self):        
        root = self.h5f.get('/')
        names = list(root.keys())
        return names
    
    def add_feature(self, species, feature):
        
        path = '{s}/{f}'.format(s=species,f=feature.name)
        group = self.h5f.require_group(path)
        group.create_dataset('')
        
    def remove(self, path):
        pass
        
    def close(self):
        pass