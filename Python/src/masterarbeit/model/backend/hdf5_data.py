#!/usr/bin/env python -W ignore::DeprecationWarning
import tables
import pandas as pd
import numpy as np
import datetime
import os
from keras.engine.topology import Container as KerasModel

from masterarbeit.model.backend.data import Data

DATETIME_FORMAT = "%H:%M:%S %d.%m.%y"

class EntriesTable(tables.IsDescription):
    index = tables.UInt64Col(pos=1)
    date = tables.StringCol(32, pos=2)
    table = tables.StringCol(32, pos=3)
    
class HDF5Pandas(Data):
    feature_path = '{species}/{feature}'
    date_column = 'datetime'
    def __init__(self):
        self.store = None
    
    def open(self, source):
        self.store = pd.get_store(source)
        
    def get_species(self):   
        # the upper folders represent the names of the species
        names = [os.path.split(k)[0].strip('/') for k in self.store.keys()]
        return np.unique(names)
    
    def add_feature(self, species, feature): 
        self.add_features(species, [feature])   
                
    def add_species(self, species):
        self.store.createGroup(self.species_root, species)
        
    def add_features(self, features, replace=False):        
        now = datetime.datetime.now().strftime(DATETIME_FORMAT)  
        features.sort(key=lambda f: type(f).__name__, reverse=True)        
        feat_type = type(None)
        for feature in features:
            species = feature.species    
            table_path = self.feature_path.format(
                species=species, feature=type(feature).__name__)          
            # type of feature changed: change paths
            if not isinstance(feature, feat_type):   
                feat_type = type(feature)
                columns = feature.columns      
                if replace and table_path in self.store:
                    del(self.store[table_path])
                                             
            now = datetime.datetime.now().strftime(DATETIME_FORMAT)
            df = pd.DataFrame([feature.values], columns=columns)
            df[self.date_column] = now
            self.store.append(table_path, df) 
            
    def commit(self):
        self.store.flush(fsync=True)
            
    def read_features(self, cls, species=None):        
        av_species = self.get_species()
        features = pd.DataFrame([], columns=cls.columns + ['species'])
        for s in av_species:
            if species is not None and s not in species:
                continue
            table_path = self.feature_path.format(
                species=s, feature=cls.__name__)   
            df = self.store.get(table_path)
            df['species'] = s
            features = features.append(df)
        if self.date_column in features:    
            del features[self.date_column]        
        return features
    
    def save_classifier(self, classifier):
        if isinstance(classifier.model, KerasModel):
            self._pickle_and_save_model(classifier.model, 'test_fasaf.h5')
            
    def load_classifier(self, name, cls):
        pass
    
    def _save_keras_model(self, model, filename):
        model.save(filename)                
    
    def _pickle_and_save_model(self, model, filename):
        pass
    
    def close(self):        
        self.store.close()

class HDF5Tables(Data):    
    '''
    store the features in seperate tables with strong hierarchical group-order, 
    makes only use of PyTables    
    easy to read
    '''
    species_root = '/'
    entries_table = 'entries'
    feature_table_name = 'feature_{index}'
    
    def __init__(self):
        self.store = None
    
    def open(self, source):
        self.store = tables.open_file(source, mode='a')
        
    def get_species(self): 
        # the upper folders represent the names of the species
        names = [g._v_name for g in self.store.list_nodes(self.species_root)]
        return names
            
    def add_feature(self, species, feature): 
        self.add_features(species, [feature])
        
    def _get_entries(self, species_group, feat_name):
        # create feature path, if not exists
        if feat_name not in species_group:
            feat_group = self.store.create_group(species_group, 
                                                 feat_name)
            # create the lookup table for added entries as well
            entries = self.store.create_table(
                feat_group, self.entries_table,
                description=EntriesTable,
                title='lookup table for all added features'
            )
        else:
            feat_group = self.store.get_node(species_group, 
                                             feat_name)                      
            entries = self.store.get_node(feat_group, self.entries_table)
        # manual indexing(alternative: pytables autoindex)
        # get last used index for type of feature 
        # (0, if no entries yet)                    
        indices = entries.col('index')
        if len(indices > 0):
            index = int(indices.max() + 1)
        else:
            index = 0 
        
        return feat_group, entries, index
        
    
    def add_features(self, species, features): 
        try:
            species_group = self.store.get_node(self.species_root, species)
        except:
            self.store.createGroup(self.species_root, species)
        feat_name = None        
        for feature in features:
            # type of feature changed: change paths
            if feat_name != feature.name:             
                feat_name = feature.name  
                feat_group, entries, index = self._get_entries(species_group, 
                                                               feat_name)                    
            entry = entries.row                    
            now = datetime.datetime.now().strftime(DATETIME_FORMAT)                
            entry['index'] = index
            entry['date'] = now
            feature_table = self.feature_table_name.format(index=index)
            entry['table'] = feature_table
            entry.append()
            entries.flush()
            self.store.create_array(feat_group, feature_table, feature.values)
            index += 1
        
    def close(self):        
        self.store.close()