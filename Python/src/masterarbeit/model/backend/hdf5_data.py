#!/usr/bin/env python -W ignore::DeprecationWarning

from masterarbeit.model.backend.data import Data
import tables
import numpy as np
import datetime
import os

DATETIME_FORMAT = "%H:%M:%S %d.%m.%y"

class EntriesTable(tables.IsDescription):
    index = tables.UInt64Col(pos=1)
    date = tables.StringCol(32, pos=2)
    table = tables.StringCol(32, pos=3)

class HDF5Data(Data):    
    
    species_root = '/'
    entries_table = 'entries'
    feature_table_name = 'feature_{index}'
    #entries_path = '{species}/{feature}/entries'
    #feature_table_path = '{species}/{feature}/{entry}'
    
    def __init__(self):
        self.store = None
    
    def open(self, source):
        self.store = tables.open_file(source, mode='a')
        
    def get_species(self): 
        # the upper folders represent the names of the species
        names = [g._v_name for g in self.store.list_nodes(self.species_root)]
        return names
    
    def add_species(self, species):
        self.store.createGroup(self.species_root, species)
        
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
        species_group = self.store.get_node(self.species_root, species)
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