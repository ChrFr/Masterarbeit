import seaborn as sns
import pandas as pd
import numpy as np

from masterarbeit.model.backend.hdf5_data import HDF5Pandas
from masterarbeit.model.features.hu_moments import HuMoments
from masterarbeit.model.features.common import features_to_dataframe

def pairplot(features, max_dim=None):
    sns.set(style="ticks", color_codes=True)      
    feature_frame = features_to_dataframe(features)        
    #species = feature_frame['species']
    #del feature_frame['species']
    #feature_frame = (feature_frame - feature_frame.mean()) / (feature_frame.max() - feature_frame.min())
    #feature_frame['species'] = species
    
    col_count = len(feature_frame.columns) - 1    
    columns = np.array(feature_frame.columns)
    if max_dim is None:
        max_dim = col_count
    for i in np.arange(max_dim, col_count + 1, max_dim):
        end = min(i, col_count)
        slc = np.arange(i - max_dim, end)
        sliced_columns = np.append(np.array(feature_frame.columns[slc]), 
                                   ['category'])
        sliced_frame = feature_frame[sliced_columns]
        g = sns.pairplot(sliced_frame, hue='category')
        sns.plt.show()
