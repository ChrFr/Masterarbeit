import numpy as np
from abc import ABCMeta
from abc import abstractmethod

class Classifier(metaclass=ABCMeta):   
    
    label = 'None'    
    columns = []      
    
    def __init__(self, name):
        self.name = name
        
    @abstractmethod
    def setup(self, input_dim):
        pass
    
    @abstractmethod        
    def train(self, image):
        pass
    
    @abstractmethod
    def load(self, source):
        pass
    
    @abstractmethod
    def save(self, source):
        pass
    
    @abstractmethod
    def predict(self, features):
        pass
    
    
if __name__ == '__main__':    
    pass