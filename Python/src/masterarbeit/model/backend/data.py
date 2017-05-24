'''
contains the abstract class for storing and accessing features, codebooks and 
classifiers

(c) 2017, Christoph Franke

this file is part of the master thesis 
"Computergestuetzte Identifikation von Pflanzen anhand ihrer Blattmerkmale"
'''
__author__ = "Christoph Franke"

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
    def list_categories(self):        
        pass    
    
    @abstractmethod    
    def add_features(self, category, feature):
        pass
    
    @abstractmethod
    def close(self):
        pass        
    def open(self, source):
        self.source = source
        self.store = pd.get_store(source)
    
        @property
        def is_open(self):
            return self.store.is_open
    
        def list_categories(self):   
            # use pytables for better hierarchical representation than pandas
            store = tables.open_file(self.source, mode='a')
            # the direct subfolders of /feature represent the names of the category
            try:
                names = [g._v_name for g in store.list_nodes(self.root + 
                                                             self.feature_base_path)]
            except:
                names = [] 
            finally:
                store.close()
            return names
    
        def get_feature_count(self, category, feature_type):
            path = self.feature_path.format(category=category, 
                                            feature=feature_type.__name__)
            if path not in self.store:
                return 0
            df = self.store[path]
            if df is None:
                return 0
            return len(df.index)        
    
        def add_feature(self, category, feature): 
            self._add_features_to_category([feature], category)   
    
        def add_category(self, category):
            self.store.createGroup(self.category_root, category)
    
        def add_features(self, features, category=None, replace=False):
            """add features to store

            request features from store by type of feature
        
            Args:
                features: list, features to be stored
                category(optional): str, category the will be assigned to, if
                                    not given, they are taken from the feats
                codebook_type(optional): if raw feature is stored, it will be 
                                         transformed by given type of codebook,
                                         codebook has to be in store
                samples_per_category(optional): int, take only n samples of each
                                                category
            
        
            Returns:
                list: a list of requested features

            """
            pass
    
        def delete_category(self, category):           
            """delete category

            delete all features of given category
        
            Args:
                category: string, the category of the features to be deleted            
            """      
            pass
        
        @abstract    
        def delete_feature_type(self, feature_type):            
            """delete features

            delete all features of given type
        
            Args:
                feature_type: the type of the features to be deleted            
            """       
            pass
                    
        @abstract       
        def commit(self):
            """commit all changes
            """
            pass
        
        @abstract    
        def get_features(self, feature_type, categories=None, codebook_type=None,
                         samples_per_category=None):
            """get stored features

            request features from store by type of feature
        
            Args:
                feature_type: the type of the features to be returned
                categories(optional): list, request spec. categories only
                codebook_type(optional): if raw feature is stored, it will be 
                                         transformed by given type of codebook,
                                         codebook has to be in store
                samples_per_category(optional): int, take only n samples of each
                                                category
            
        
            Returns:
                list: a list of requested features

            """
            pass
    
        @abstract
        def get_joined_features(self, feat_types, categories=None,
                                codebook_type=None):
            pass
        
        @abstract     
        def save_classifier(self, classifier):
            pass
                            
        @abstract      
        def list_classifiers(self):
            pass
        
        @abstract     
        def get_classifier(self, cls, name):
            pass
        
        @abstract 
        def delete_classifier(self, cls, name):
            pass
            
        @abstract        
        def save_codebook(self, codebook, feature_type):
            pass
            
        @abstract    
        def get_codebook(self, feature_type, codebook_type):
            pass
    
        @abstract
        def list_codebooks(self, feature_type):   
            pass
    
def class_to_string(cls):
    if cls is None:
        return ''
    mod = cls.__module__
    cls_name = cls.__name__
    return '{mod}.{cls}'.format(mod=mod, cls=cls_name)

def load_class(class_string):
    if not class_string:
        return None
    split = class_string.split('.')
    module_str = '.'.join(split[:-1])
    module = importlib.import_module(module_str)
    cls_str = split[-1]    
    cls = getattr(module, cls_str)
    return cls