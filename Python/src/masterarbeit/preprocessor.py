from abc import ABC, abstractmethod

class PreProcessor(ABC):
    processed_pixels = None
    source_pixels = None
    
    @abstractmethod    
    def read_file(self, filename):     
        pass
    
    @abstractmethod    
    def crop(self, filename):     
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
    
    @abstractmethod
    def write_to(self):
        pass
    
    def get_source_image(self):
        return self.source_pixels
    
    def get_processed_image(self):
        return self.processed_pixels