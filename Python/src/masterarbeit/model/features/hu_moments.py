import cv2
from masterarbeit.model.features.feature import Feature

class HuMoments():
    label = 'Hu-Moments'
    
    def describe(self, binary):
        moments = cv2.HuMoments(cv2.moments(binary))
        feature = Feature(self.label)
        feature.values = moments
        return feature
    
    