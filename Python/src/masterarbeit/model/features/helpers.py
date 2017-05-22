'''
contains auxiliary functions for feature handling

(c) 2017, Christoph Franke

this file is part of the master thesis 
"Computergestuetzte Identifikation von Pflanzen anhand ihrer Blattmerkmale"
'''
__author__ = "Christoph Franke"

import pandas as pd

def features_to_dataframe(features):
    if len(features) == 0:
        return []    
    columns = features[0].columns   
    feat_type = type(features[0])
    dataframe = None
    for feature in features:
        if type(feature) != feat_type:
            raise Exception(
                'Only one type feature allowed in coversion to dataframe!')
        category = feature.category           
        df = pd.DataFrame([feature.values], columns=columns) 
        df['category'] = category
        if dataframe is None:
            dataframe = df
        else:
            dataframe = dataframe.append(df)     
    return dataframe