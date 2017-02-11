#!/usr/bin/env python -W ignore::DeprecationWarning
import tables
import pandas as pd
import numpy as np
import datetime
import os
import tempfile
import uuid
import shutil
import json
from collections import OrderedDict
from keras.engine.topology import Container as KerasModel
from keras.models import load_model as keras_load_model
import pickle

from masterarbeit.model.backend.data import (Data, class_to_string, 
                                             load_class)

DATETIME_FORMAT = "%H:%M:%S %d.%m.%y"

class EntriesTable(tables.IsDescription):
    index = tables.UInt64Col(pos=1)
    date = tables.StringCol(32, pos=2)
    table = tables.StringCol(32, pos=3)   
    
class HDF5Pandas(Data):
    root = '/'
    classifier_config_attr = 'classifier_config'
    serialized_model_attr = 'classifier_serialized'
    feature_base_path = 'features'
    model_base_path = 'models'
    dictionary_base_path = 'dictionaries'
    category_path = feature_base_path + '/{category}'
    feature_path = category_path + '/{feature}'
    codebook_path = dictionary_base_path + '/{feature}'
    model_path = model_base_path + '/{model}/{name}'
    date_column = 'datetime'
    
    def __init__(self):
        self.store = None
        self.source = None
    
    def open(self, source):
        self.source = source
        self.store = pd.get_store(source)
        
    @property
    def is_open(self):
        return self.store.is_open
        
    def get_categories(self):   
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
        path = self.feature_path.format(category=category, feature=feature_type)
        if path not in self.store:
            return 0
        df = self.store[path]
        return len(df.index)        
    
    def add_feature(self, category, feature): 
        self.add_features(category, [feature])   
                
    def add_category(self, category):
        self.store.createGroup(self.category_root, category)
        
    def add_features(self, features, replace=False):
        features = np.array(features)
        categories = np.array([f.category for f in features])
        unique_cat = np.unique(categories)
        for category in unique_cat:
            idx = categories == category
            self._add_features(features[idx], category, replace=replace)        
        
    def _add_features(self, features, category, replace=False):        
        now = datetime.datetime.now().strftime(DATETIME_FORMAT)    
        feat_types = np.array([type(f).__name__ for f in features])  
        unique_feat = np.unique(feat_types)
        for feat_type in feat_types:  
            table_path = self.feature_path.format(
                category=category, feature=feat_type)  
            if replace and self.root + table_path in self.store.keys():                
                del(self.store[table_path])        
            idx = feat_types == feat_type   
            features_per_type = features[idx]
            for feature in features_per_type:
                columns = feature.columns  
                df = pd.DataFrame([feature.values], columns=columns)
                df[self.date_column] = now        
                self.store.append(table_path, df) 
        self.commit()
        
    def delete_category(self, category):
        # when pandas is used for deletion of keys, parent groups are kept
        # so pytables is used for cleaner deletion here
        store = tables.open_file(self.source, mode='a')
        category_path = self.root + self.category_path.format(category=category)
        store.remove_node(category_path, recursive=True)
        store.close()              
        
    def delete_feature_type(self, feature_type):
        for path in self.store.keys():
            if path.endswith('/' + feature_type.__name__):
                del self.store[path]
        code_path = self.codebook_path.format(feature=feature_type.__name__)  
        del self.store[code_path]
        
    def commit(self):
        self.store.flush(fsync=True)
            
    def _get_feature_frame(self, cls, categories=None):        
        av_category = self.get_categories()
        feature_frame = None
        for category in av_category:
            if categories is not None and category not in categories:
                continue
            table_path = self.feature_path.format(
                category=category, feature=cls.__name__)   
            df = self.store.get(table_path)
            df['category'] = category
            if feature_frame is None:
                feature_frame = df
            else:            
                feature_frame = feature_frame.append(df)
        if self.date_column in feature_frame:    
            del feature_frame[self.date_column]        
        return feature_frame
    
    def get_features(self, cls, categories=None):
        feature_frame = self._get_feature_frame(cls, categories=categories)
        features = []
        for row in feature_frame.iterrows():
            r = row[1]
            category = r['category']
            del r['category']
            values = np.array([v for v in r])
            feature = cls(category)  
            feature.values = values
            features.append(feature)
        return features
    
    def save_classifier(self, classifier):
        table_path = self.model_path.format(model=type(classifier).__name__,
                                            name=classifier.name)
        #self.store[table_path] = pd.DataFrame(classifier.serialize())
        #self.commit()
        
        out_store = tables.open_file(self.source, mode='a')   
        # keras provides method to save model, but only in single file
        # write file and copy it into the main source
        if isinstance(classifier.model, KerasModel):            
            tmp_dir = tempfile.mkdtemp()
            f_name = '{}.h5'.format(uuid.uuid4())
            tmp_file = os.path.join(tmp_dir, f_name)
            self._save_keras_model(classifier.model, tmp_file) 
            try:
                in_store = tables.openFile(tmp_file, mode='a')             
                self._copy_group(in_store, '/', out_store, table_path)
                # keras saves model attributes to root
                for config_attr in ['model_config', 'training_config']:
                    model_config = in_store.get_node_attr('/', config_attr)
                    out_store.set_node_attr('/' + table_path, config_attr, 
                                            model_config) 
            finally:             
                in_store.close()
                shutil.rmtree(tmp_dir)    
        classifier_config = OrderedDict()  
        classifier_config['type'] = type(classifier).__name__
        
        feat_types_ser = [class_to_string(ft) 
                          for ft in classifier.trained_features]
        now = datetime.datetime.now().strftime(DATETIME_FORMAT) 
        classifier_config['date'] = now
        classifier_config['feature types'] = feat_types_ser
        classifier_config['input dim'] = classifier.input_dim
        classifier_config['trained categories'] = list(
            classifier.trained_categories)
        out_store.set_node_attr(self.root + table_path, 
                                self.classifier_config_attr, 
                                json.dumps(classifier_config))
        out_store.close()          
            
    def get_classifiers(self):
        classifiers = {}
        # use pytables, because models were not saved as dataframes, 
        # so pandas ignore them
        store = tables.open_file(self.source, mode='a')
        try:
            for g in store.list_nodes(self.root + self.model_base_path):
                classifier = g._v_name
                classifiers[classifier] = []
                for child in g:
                    name = child._v_name
                    classifiers[classifier].append(name)
        except:
            pass
        finally:
            store.close()
        return classifiers
            
    def get_classifier(self, cls, name):
        classifier = cls(name)
        table_path = self.model_path.format(model=cls.__name__,
                                            name=name)
        #if table_path not in self.store:
            #return None
        #df = self.store.get(table_path)
        #serialized = df.as_matrix()
        ## if array was stored, no need for 2nd dim
        #if serialized.shape[1] == 1:
            #serialized = serialized[:, 0]
        #classifier.deserialize(serialized)
        #return classifier
    
        status = OrderedDict()
        # keras provides method to load modelfrom single file
        # copy model-group into file to load it
        in_store = tables.openFile(self.source, mode='a')
        if isinstance(classifier.model, KerasModel): 
            tmp_dir = tempfile.mkdtemp()
            f_name = '{}.h5'.format(uuid.uuid4())
            tmp_file = os.path.join(tmp_dir, f_name)
            try:        
                out_store = tables.open_file(tmp_file, mode='a')
                self._copy_group(in_store, table_path, out_store, '/', 
                                 keep_group=False)
                for config_attr in ['model_config', 'training_config']:
                    config = in_store.get_node_attr('/' + table_path, 
                                                    config_attr)
                    out_store.set_node_attr('/',  config_attr, config)
                    status[config_attr] = json.loads(config.astype(str))
                model = self._load_keras_model(tmp_file)  
                classifier.model = model
            except:
                print("couldn't load Keras model")
            finally:   
                out_store.close()                
                shutil.rmtree(tmp_dir) 
        classifier_config = in_store.get_node_attr(self.root + table_path, 
                                                   self.classifier_config_attr)
        cls_config = json.loads(classifier_config)
        status['classifier'] = cls_config
        feature_types = [load_class(ft) 
                         for ft in cls_config['feature types']]
        classifier.trained_features = feature_types
        classifier.trained_categories = cls_config['trained categories']
        status.move_to_end('classifier', last=False)
        in_store.close()        
        classifier.meta = status
        return classifier        
    
    def delete_classifier(self, cls, name):
        store = tables.open_file(self.source, mode='a')
        cls_path = self.root + self.model_path.format(model=cls.__name__,
                                                      name=name)
        store.remove_node(cls_path, recursive=True)
        store.close() 
                
    def save_codebook(self, codebook, feature_type):
        table_path = self.codebook_path.format(feature=feature_type.__name__)         
        self.store[table_path] = pd.DataFrame(codebook.serialize())
        self.commit()
        
    def get_codebook(self, feature_type):
        table_path = self.codebook_path.format(feature=feature_type.__name__) 
        if table_path not in self.store:
            return None
        df = self.store.get(table_path)
        serialized = df.as_matrix()
        codebook = feature_type.codebook_type()
        codebook.deserialize(serialized)
        return codebook
    
    def _copy_group(self, in_store, in_group, out_store, out_group, keep_group=True): 
        '''
        copies a hdf5 group from one file to another
        
        params
        ------
        in_file: hdf5 file to copy from
        in_group: group in in_file to be copied
        out_file: hdf5 file to copy to
        out_group: group in out_file the in_group is copied to
        keep_group: if True, the group content is copied into a group with same 
                    name, else contents are copied to out_group without keeping
                    parent group
        '''
        # root always exists
        root = '/'
        if out_group == root:                
            out_g = out_store.get_node(root)
        # group is not root -> try to access, create if not exists 
        else:
            try:
                out_g = out_store.get_node(root, out_group)    
            except:
                p, g = os.path.split(out_group)
                out_g = out_store.create_group(root + p, g, 
                                               createparents=True)
        if not in_group.startswith(root):
            in_group = root + in_group
        if keep_group:
            copy_func = in_store.copy_node
        else:
            copy_func = in_store.copy_children
        copy_func(in_group, out_g, recursive=True, overwrite=True) 
    
    def _save_keras_model(self, model, filename):        
        model.save(filename)   
        
    def _load_keras_model(self, filename):  
        model = keras_load_model(filename)
        return model
    
    def close(self):        
        self.store.close()

class HDF5Tables(Data):    
    '''
    store the features in seperate tables with strong hierarchical group-order, 
    makes only use of PyTables    
    easy to read
    '''
    category_root = '/'
    entries_table = 'entries'
    feature_table_name = 'feature_{index}'
    
    def __init__(self):
        self.store = None
    
    def open(self, source):
        self.store = tables.open_file(source, mode='a')
        
    def get_categories(self): 
        # the upper folders represent the names of the category
        names = [g._v_name for g in self.store.list_nodes(self.category_root)]
        return names
            
    def add_feature(self, category, feature): 
        self.add_features(category, [feature])
        
    def _get_entries(self, category_group, feat_name):
        # create feature path, if not exists
        if feat_name not in category_group:
            feat_group = self.store.create_group(category_group, 
                                                 feat_name)
            # create the lookup table for added entries as well
            entries = self.store.create_table(
                feat_group, self.entries_table,
                description=EntriesTable,
                title='lookup table for all added features'
            )
        else:
            feat_group = self.store.get_node(category_group, 
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
        
    
    def add_features(self, category, features): 
        try:
            category_group = self.store.get_node(self.category_root, category)
        except:
            self.store.createGroup(self.category_root, category)
        feat_name = None        
        for feature in features:
            # type of feature changed: change paths
            if feat_name != feature.name:             
                feat_name = feature.name  
                feat_group, entries, index = self._get_entries(category_group, 
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
        
def split_list(lst, field_name):
    lst.sort(key=lambda element: geattr(element, field_name))
    ret = []
    field_val = None
    segment = []
    for element in lst:
        if field_val != geattr(element, field_name):
            if len(feat_cat_list) > 0:
                self._add_features(feat_cat_list, category, replace=replace)
            field_val = geattr(element, field_name)
            segment = [] 
        segment.append(element)     