import numpy as np
import os
import cv2

from masterarbeit.model.preprocessor.preprocessor import PreProcessor

class OpenCVPreProcessor(PreProcessor):
    
    @staticmethod
    def read(filename):     
        """
        read image from file
        
        Parameters
        ----------
        filename : str, name of the file to read       
        """           
        pixel_array = cv2.imread(filename, cv2.IMREAD_COLOR) 
        cv2.cvtColor(pixel_array, cv2.COLOR_BGR2RGB, pixel_array)
        return pixel_array    
    
    @staticmethod    
    def write(image, filename):
        pixels = image.copy()
        cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR, pixels)
        success = cv2.imwrite(filename, pixels)
        return success
        

class Binarize(OpenCVPreProcessor):
    
    def __init__(self):
        self.source_pixels = None
        self.processed_pixels = None
    
    def _mask(image, mask):
        masked_source = image.copy()
        
        # colored (3 values per pixel)
        if len(masked_source.shape) == 3:
            for i in range(3):
                masked_source[:, :, i] = np.multiply(masked_source[:, :, i], mask)
        # black/white or grey
        else:
            masked_source = np.multiply(masked_source, masked_source) 
            
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
        
        gabor = gabor(grey)
        segmented = cv2.adaptiveThreshold(gabor, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 6) 
        return gabor    
    
    def segment(self):
        return self.source_pixels
    
    @staticmethod      
    def process(image, steps_dict=None):
        binary, mask = binarize(image)
        #masked_source = self._mask(image)
        if steps_dict is not None:
            steps_dict['binary'] = binary
            #steps_dict['masked source'] = masked_source    
        #cropped = scale_to_bounding_box(binary.copy(), masked_source)
        return binary
    
    def crop(image):
        binary, mask = binarize(image)
        masked_source = Binarize._mask(image, mask)        
        cropped = scale_to_bounding_box(binary, masked_source)     
        return cropped
        
    def segment_veins(self):
        
        grey = cv2.cvtColor(self.source_pixels, cv2.COLOR_RGB2GRAY)   
        gabor = gabor(grey)           
        
        #lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        #lines = lsd.detect(gabor)
        #line_img = lsd.drawSegments(np.empty(grey.shape), lines[0])    
        #cv2.imshow('', line_img)        
                
        thresh = self._mask(cv2.threshold(gabor, 75, 255, cv2.THRESH_BINARY)[1])
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations = 5)
        dilation = cv2.dilate(erosion, kernel, iterations = 5)  
        
    
    
def binarize(image):
    #self.processed_pixels = cv2.GaussianBlur(self.processed_pixels, (5,5), 0)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #grey = cv2.equalizeHist(grey)
    #blur = cv2.GaussianBlur(grey, (11, 11), 0)
    ret, binary = cv2.threshold(grey, 220, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)         
    #binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 6)    
    #binarized = cv2.Canny( grey, 1, 200, 100 );

    #binarized = cv2.pyrMeanShiftFiltering(self.source_pixels, 30, 30, 3);

    #green = self.source_pixels[:, :, 1]
    #ret, binary = cv2.threshold(green, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)     
    binary_mask = np.clip(binary, 0, 1)   
    return binary, binary_mask    

def scale_to_bounding_box(binary, image):
    contours = cv2.findContours(binary, 1, 2)
    x,y,w,h = cv2.boundingRect(contours[0])
    cropped = image[y: y + h, x: x + w]
    return cropped.copy()    

def gabor(image):

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