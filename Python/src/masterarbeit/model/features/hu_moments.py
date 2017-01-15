import cv2
from masterarbeit.model.features.feature import Feature

class HuMoments():
    feature_name = 'Hu-moments'
    
    def describe(self, binary):
        moments = cv2.HuMoments(cv2.moments(binary))
        feature = Feature(self.feature_name)
        feature.values = moments
        return feature
    
    