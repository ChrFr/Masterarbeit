'''
contains functions to plot feature vectors

(c) 2017, Christoph Franke

this file is part of the master thesis 
"Computergestuetzte Identifikation von Pflanzen anhand ihrer Blattmerkmale"
'''
__author__ = "Christoph Franke"

import seaborn as sns
import pandas as pd
import numpy as np
from masterarbeit.model.features.helpers import features_to_dataframe

def pairplot(features, max_dim=None, do_show=False):
    '''plot pairwise distribution of feature values

    Args:
        max_dim (optional): int, maximum dimension per plot, 
                            if not given all dimension are plotted in one vis.
        do_show (optional): bool, if True, visualisation is shown when calling
                            this function, else only drawn to canvas

    '''
    sns.set(style="ticks", color_codes=True)      
    feature_frame = features_to_dataframe(features)        
    
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
        grid = sns.pairplot(sliced_frame, hue='category')
        
        if do_show:
            sns.plt.show()
        else:
            sns.plt.draw()
            
