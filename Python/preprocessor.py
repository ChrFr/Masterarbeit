from abc import ABC, abstractmethod

class PreProcessor(ABC):
    @abstractmethod    
    def read_file(self, filename):     
        pass
    
    @abstractmethod
    def binarize(self):
        pass
    
    @abstractmethod
    def segment(self):
        pass
    
    @abstractmethod
    def process(self):
        pass