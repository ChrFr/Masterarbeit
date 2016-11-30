import cv2
import numpy as np
from collections import OrderedDict
import os

from preprocessor import PreProcessor

class OpenCVPreProcessor(PreProcessor):
    
    def __init__(self):
        self.source_pixels = None
        self.processed_images = OrderedDict()
    
    def read_file(self, filename):     
        """
        read image from file
        
        Parameters
        ----------
        filename : str, name of the file to read       
        """           
        pixel_array = cv2.imread(filename, cv2.IMREAD_COLOR) 
        if pixel_array is None:
            return
        cv2.cvtColor(pixel_array, cv2.COLOR_BGR2RGB, pixel_array)
        self.source_pixels = pixel_array  
        return self.source_pixels
            
    def binarize(self):
        grey = cv2.cvtColor(self.source_pixels, cv2.COLOR_BGR2GRAY)
        #blur = cv2.GaussianBlur(grey, (5, 5), 0)
        ret, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)         
        #binarized = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 6)    
        #binarized = cv2.Canny( grey, 1, 200, 100 );
               
        #binarized = cv2.pyrMeanShiftFiltering(self.source_pixels, 30, 30, 3);
        
        #green = self.source_pixels[:, :, 1]
        #ret, binarized = cv2.threshold(green, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)     
        self.binary_mask = np.clip(binary, 0, 1)      
        return binary
    
    def _mask(self, source):
        masked_source = source.copy()
        
        # colored (3 values per pixel)
        if len(source.shape) == 3:
            for i in range(3):
                masked_source[:, :, i] = np.multiply(masked_source[:, :, i], self.binary_mask)
        # black/white or grey
        else:
            masked_source = np.multiply(masked_source, self.binary_mask) 
            
        return masked_source
    
    def _segment_veins(self):
        
        grey = cv2.cvtColor(self.source_pixels, cv2.COLOR_RGB2GRAY)        
        
        #hsv = cv2.cvtColor(self.source_pixels, cv2.COLOR_RGB2HSV)
        #h = hsv[:, :, 0]  
        #v = hsv[:, :, 2]
        #y = (((h + 90) % 360) / 360 + 1 - v / 255 ) / 2 * 255
        
        
        #grey = cv2.GaussianBlur(grey,(5,5),0)
       # segmented = cv2.Canny(h, 15, 30, 3)
        
        #lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        #lines = lsd.detect(grey)
        #segmented = lsd.drawSegments(np.empty(grey.shape), lines[0])
        
        #segmented = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 6)        
        #kernel = np.ones((2,2),np.uint8)
        #erosion = cv2.erode(segmented, kernel, iterations = 1)
        #dilation = cv2.dilate(erosion, kernel, iterations = 1)
        
        gabor = _gabor(grey)
        segmented = cv2.adaptiveThreshold(gabor, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 6) 
        return gabor
    
    
    def segment(self):
        return self.source_pixels
    
    def process(self):
        self.processed_images.clear()
        if self.source_pixels is None:
            return
        binary = self.binarize()
        masked_source = self._mask(self.source_pixels)
        
        processed_images['binary'] = binary
        processed_images['masked source'] = masked_source      
        
        grey = cv2.cvtColor(self.source_pixels, cv2.COLOR_RGB2GRAY)   
        gabor = _gabor(grey)
        processed_images['gabor'] = gabor        
        
        #lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        #lines = lsd.detect(gabor)
        #line_img = lsd.drawSegments(np.empty(grey.shape), lines[0])    
        #processed_images['lines'] = line_img   
        #cv2.imshow('', line_img)
        
        #ret, binary = cv2.threshold(gabor, 75, 255, cv2.THRESH_BINARY)  
        #processed_images['gabor binary'] = binary
        
        #adaptive = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 6)   
        #processed_images['adaptive'] = adaptive      
        
        thresh = self._mask(cv2.threshold(gabor, 60, 255, cv2.THRESH_BINARY)[1])
        processed_images['thresh'] = thresh      
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations = 3)
        dilation = cv2.dilate(erosion, kernel, iterations = 3)        
        processed_images['opening'] = dilation     
        processed_images['removed opening'] = thresh - dilation         
        #processed_images['closing'] = dilation
        
        #processed_images['gabor_thresh'] = cv2.adaptiveThreshold(gabor, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 6)        

def _gabor(image):

    def build_filters():
        filters = []
        ksize = 31
        for theta in np.arange(0, np.pi, np.pi / 16):
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
        return filters

    def process(img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum        

    filters = build_filters()

    res1 = process(image, filters)

    return res1        