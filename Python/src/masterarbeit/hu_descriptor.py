import cv2
from descriptor import Descriptor

class HuDescriptor(Descriptor):
    def __init__(self):
        pass
    
    def describe(self, binary):
        moments = cv2.HuMoments(cv2.moments(binary))
        return moments.flatten()