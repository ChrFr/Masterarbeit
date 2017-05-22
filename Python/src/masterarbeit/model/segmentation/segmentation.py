'''
contains the abstract class and Implementations for segmenting images

(c) 2017, Christoph Franke

this file is part of the master thesis 
"Computergestuetzte Identifikation von Pflanzen anhand ihrer Blattmerkmale"
'''
__author__ = "Christoph Franke"

from abc import ABCMeta
from abc import abstractmethod
import math
import numpy as np
import cv2
import math
from sklearn.cluster import KMeans  

from masterarbeit.model.segmentation.helpers import (mask, crop, 
                                                     simple_binarize,
                                                     remove_small_objects,
                                                     fill_holes)
class Segmentation(metaclass=ABCMeta):  
    
    label = 'None'   
    n_pixels = 2000000
    
    def process(self, image, steps=None):
        binary = self._process(image, steps=steps)
        if binary.sum() == 0:
            return image
        cropped = crop(mask(image, binary))
        return cropped
         
    @abstractmethod
    def _process(self, image, steps=None):
        return
    
    def _common_scale(self, image, downscale=1):
        resolution = image.shape[0] * image.shape[1]
        scale = math.sqrt(self.n_pixels / resolution) / downscale       
        new_shape = (np.array(image.shape[:2]) * scale).astype(np.int)
        # cv2 swaps width with height
        scaled = cv2.resize(image, (new_shape[1], new_shape[0]))
        return scaled        
    
    
class Binarize(Segmentation):    
    
    label = 'Binarize image with Otsu-Thresholding'        
    
    def _process(self, image, steps=None):        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)        
        ret, binary = cv2.threshold(image, 220, 255, 
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 
        
        binary = np.clip(binary, 0, 1)
                
        # sometimes are in different order as expected
        # label 0 is supposed to be our background
        border_sum = (binary[1, :].sum() + 
                      binary[-1, :].sum() + 
                      binary[:, 1].sum() + 
                      binary[:, -1].sum())
        
        # if the border labels if they have label 1 by majority -> revert
        if border_sum > (image.shape[0] + image.shape[1]):
            binary = 1 - binary
                    
        if steps is not None:
            steps['binary'] = binary
        return binary
    
    
class BinarizeHSV(Binarize):     
    
    label = 'Binarize image with Otsu-Thresholding in HSV space' 
    
    def _process(self, image, steps=None):
        resized = self._common_scale(image)     
        
        hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)    
        # take saturation only, remove all others
        hsv_image[:, :, 0] = 0
        hsv_image[:, :, 2] = 0
        res_binary = super(BinarizeHSV, self)._process(hsv_image)
        # fill holes in leaf (caused by reflections or actual holes)
        wo_holes = fill_holes(res_binary)
                
        if steps is not None:
            steps['saturation'] = hsv_image
            steps['binary'] = res_binary
            steps['binary without holes'] = wo_holes
            
        rem = remove_small_objects(wo_holes, 10000)
        o_sized = cv2.resize(rem, (image.shape[1], image.shape[0]))
        return o_sized.astype(np.uint8)
         
    
class KMeansBinarize(Segmentation):
    label = 'clustered k-means Binarization'     
    
    def _process(self, image, steps=None):
        resized = self._common_scale(image)       
        
        reshaped_img = resized.reshape((-1,3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10)
        # for binarization we need the color space clustered into 2 labels
        n_clusters = 2
              
        #ret, label, center = cv2.kmeans(reshaped_img, n_clusters, None,
            #criteria, 1, cv2.KMEANS_PP_CENTERS)    
        #if steps is not None:   
            #res = center[label.flatten()]
            #res2 = res.reshape((resized.shape))
            #steps['clustered colors'] = res2   
            
        # kmeans of sklearn works better than cv2 and outcome is more predictable
        kmeans = KMeans(n_clusters=n_clusters)
        label = kmeans.fit_predict(reshaped_img)   
                        
        height, width = resized.shape[:2]
        labels_reshaped = label.reshape(
            (height, width)).astype(np.float32)      
        
        # label 0 is supposed to be the background, check the borders 
        border_sum = (labels_reshaped[1, :].sum() + 
                      labels_reshaped[-1, :].sum() + 
                      labels_reshaped[:, 1].sum() + 
                      labels_reshaped[:, -1].sum())
        
        # if the border labels have label 1 by majority -> revert
        if border_sum > (width + height):
            labels_reshaped = 1 - labels_reshaped
            
        # remove noise with gauss-filter (incl. thresholding, gauss 
        # creates iterim values between 0 and 255)       
        labels_reshaped = cv2.GaussianBlur(labels_reshaped, (0, 0), 2)     
        retval, binary = cv2.threshold(labels_reshaped, .5, 255.0, 
                                       cv2.THRESH_BINARY)
        
        wo_holes = fill_holes(binary)
        if steps is not None:
            steps['wo'] = wo_holes
            
        rem = remove_small_objects(wo_holes, 10000)
        rem = rem.astype(np.uint8)
        o_sized = cv2.resize(rem, (image.shape[1], image.shape[0]))
        
        return o_sized
    
    
class KMeansHSVBinarize(KMeansBinarize):
    label = 'clustered k-means Binarization in HSV space'   
    
    def _process(self, image, steps=None):
        hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
        # remove hue
        hsv_image[:, :, 0] = 0
        if steps is not None:
            steps['hsv without hue'] = hsv_image
        binary = super(KMeansHSVBinarize, self)._process(hsv_image, steps)

        return binary
    
    
class Slic(Segmentation):    
    
    label = ('Superpixels, grouped by color' +
             '(center of image is binarized)')
    
    def _process(self, image, steps=None):    
        from skimage import segmentation, color 
        from skimage.future import graph
        
        resized = self._common_scale(image, downscale=5)
        labeled = segmentation.slic(resized, compactness=30, n_segments=1000)
        superpixels = color.label2rgb(labeled, resized, kind='avg')
        g = graph.rag_mean_color(resized, labeled)
        merged_labels = graph.cut_threshold(labeled, g, 30)
        merged_superpixels = color.label2rgb(merged_labels, resized, kind='avg')
        
        if steps is not None:
            steps['superpixels'] = superpixels
            steps['merged superpixels'] = merged_superpixels
            
        center = merged_labels[int(merged_labels.shape[0] / 2), 
                               int(merged_labels.shape[1] / 2)]
        binary = np.zeros(merged_labels.shape[:2])
        binary[merged_labels == center] = 1
        binary = cv2.resize(binary, (image.shape[1], image.shape[0]))
            
        return binary
