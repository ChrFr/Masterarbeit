#import cv2
from skimage import measure #import moments, moments_central, moments_hu
from masterarbeit.model.features.feature import Feature

class HuMoments(Feature):
    label = 'Hu-Moments'
    columns = ['0', '1', '2', '3', '4', '5', '6']
    
    def describe(self, binary, steps={}):
        #moments = cv2.HuMoments(cv2.moments(binary))
        # in case range is [0,255]
        clipped = binary.clip(max=1)
        m = measure.moments(clipped)
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]
        central = measure.moments_central(clipped, cr, cc)
        normalized = measure.moments_normalized(central)
        hu = measure.moments_hu(normalized)
        self.values = hu
    
    