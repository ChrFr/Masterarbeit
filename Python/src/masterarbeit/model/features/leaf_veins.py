import cv2
from masterarbeit.model.features.feature import Feature

class LeafVeins(Feature):
    label = 'leaf veins'
    columns = ['a', 'b', 'c']
    
    def extract(self, binary):
        pass
    
    