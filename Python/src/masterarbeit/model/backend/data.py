from abc import ABC, abstractmethod

class Data(ABC):
    
    @abstractmethod
    def open(source):
        pass
    
    @abstractmethod
    def add_feature(path, array):
        pass
    
    @abstractmethod        
    def get_species(self):        
        pass    
    
    @abstractmethod    
    def add_feature(self, species, feature):
        pass
    
    @abstractmethod
    def close(self):
        pass    