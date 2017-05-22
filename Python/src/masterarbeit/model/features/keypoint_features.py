'''
contains features to describe the keypoints of colored images

(c) 2017, Christoph Franke

this file is part of the master thesis 
"Computergestuetzte Identifikation von Pflanzen anhand ihrer Blattmerkmale"
'''
__author__ = "Christoph Franke"

import cv2
import numpy as np

from masterarbeit.model.features.feature import UnorderedFeature

class Sift(UnorderedFeature):
    label = 'Sift Keypoints'
    histogram_length = 300
    
    def _describe(self, image, steps=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
        gray[gray==255] = 0
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, des = sift.detectAndCompute(gray, None)
        
        if steps is not None:
            img_keys = cv2.drawKeypoints(
                image, keypoints, None, 
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            steps['keypoints'] = img_keys
            
        return des
           
    
class Surf(UnorderedFeature):
    label = 'Surf Keypoints'
    histogram_length = 300
    
    def _describe(self, image, steps=None):    
        surf = cv2.xfeatures2d.SURF_create()
        keypoints, des = surf.detectAndCompute(image, None)
        
        if steps is not None:
            img_keys = cv2.drawKeypoints(
                image, keypoints, None, 
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            steps['keypoints'] = img_keys
            
        return des
    