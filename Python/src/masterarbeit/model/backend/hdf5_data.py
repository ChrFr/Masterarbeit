#!/usr/bin/env python -W ignore::DeprecationWarning

from masterarbeit.model.backend.data import Data
import pandas as pd
import numpy as np
import datetime
import os

class HDF5Data(Data):
    
    entries_path = '{species}/{feature}/entries'
    feature_table_name = 'feature_table_{index}'
    feature_table_path = '{species}/{feature}/{entry}'
    
    def __init__(self):
        self.store = None
    
    def open(self, source):
        self.store = pd.get_store(source, mode='a')
        
    def get_species(self):   
        # the upper folders represent the names of the species
        names = [k.split('/')[1] for k in self.store.keys()]
        return names
    
    def add_feature(self, species, feature):
        entries_path = self.entries_path.format(species=species,
                                                feature=feature.name)
        if self.store.get_node(entries_path) is None:
            columns = ['datetime_added', 'entry_table']
            entries = pd.DataFrame(columns=columns)
        else:
            entries = self.store.get(entries_path)
            
        # get index for new entry (0, if no entries yet; else increment inde)
        indices = entries.index
        # no entries yet
        if len(indices) == 0:
            index = 0
        # increment index
        else:
            index = indices.max() + 1
            
        feature_table = self.feature_table_name.format(index=index)
        now = datetime.datetime.now().strftime("%H:%M:%S-%d.%m.%y")
        entries.loc[index] = [now, feature_table]
        self.store[entries_path] = entries
        table_path = self.feature_table_path.format(species=species,
                                                    feature=feature.name,
                                                    entry=feature_table)
        #self.store[table_path] = 
        ##df = pd.DataFrame(, index=[now])
        #df = pd.DataFrame([feature.values], columns=np.arange(len(feature.values)), index=[now])      
        #self.store.append(path, df)
        
    def close(self):        
        self.store.close()