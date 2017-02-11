from abc import ABC, abstractmethod
import importlib

class Data(ABC):
    
    @abstractmethod
    def open(source):
        pass
    
    @abstractmethod
    def add_feature(path, array):
        pass
    
    @abstractmethod        
    def get_categories(self):        
        pass    
    
    @abstractmethod    
    def add_features(self, category, feature):
        pass
    
    @abstractmethod
    def close(self):
        pass        
    
def class_to_string(cls):
    mod = cls.__module__
    cls_name = cls.__name__
    return '{mod}.{cls}'.format(mod=mod, cls=cls_name)

def load_class(class_string):
    split = class_string.split('.')
    module_str = '.'.join(split[:-1])
    module = importlib.import_module(module_str)
    cls_str = split[-1]    
    cls = getattr(module, cls_str)
    return cls