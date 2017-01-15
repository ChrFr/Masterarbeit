from abc import ABC, abstractmethod

class PreProcessor(ABC):
    
    @abstractmethod    
    def read(path):
        pass
    
    @abstractmethod    
    def write(path):
        pass
    
    @abstractmethod    
    def process(image):
        pass