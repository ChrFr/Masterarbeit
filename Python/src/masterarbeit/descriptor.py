from abc import ABC, abstractmethod

class Descriptor(ABC):
    @abstractmethod    
    def describe(self, binary):
        pass