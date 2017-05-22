'''
contains auxiliary functions for reading and segmenting files

(c) 2017, Christoph Franke

this file is part of the master thesis 
"Computergestuetzte Identifikation von Pflanzen anhand ihrer Blattmerkmale"
'''
__author__ = "Christoph Franke"

import numpy as np
import cv2
   
def simple_binarize(segmented_image):
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    gray[gray >= 230] = 0
    binary = np.clip(gray, 0, 1)  
    return binary

def remove_small_objects(binary, min_area): 
    binary = binary.copy()
    i, contours, h = cv2.findContours(binary, cv2.RETR_TREE, 
                                      cv2.CHAIN_APPROX_SIMPLE) 
    for contour in contours:
        area = cv2.contourArea(contour);
        if (abs(area) < min_area):
            binary = cv2.drawContours(binary, [contour], 0, 0, 
                                      thickness=cv2.FILLED)            
    return binary

def fill_holes(binary):
    filled = binary.copy().astype(np.uint8)
    mask = np.zeros((filled.shape[0]+2, filled.shape[1]+2), np.uint8)
    # that value should not appear in a binary image (0, 255 or 1)
    fill_code = 75
    # fill image
    cv2.floodFill(filled, mask, (0,0), fill_code)
    # everything that was not filled belongs to the object
    filled[filled != fill_code] = binary.max()
    filled[filled == fill_code] = 0
    return filled

def remove_thin_objects(segmented_image):  
    binary = simple_binarize(segmented_image)
    process_width = 800
    res_factor = process_width / binary.shape[1]
    new_shape = (int(binary.shape[1] * res_factor), 
                 int(binary.shape[0] * res_factor))
    resized = cv2.resize(binary, new_shape)
    disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  
    eroded = cv2.erode(resized, disk_kernel)
        
    # dilate with much bigger structuring element ('blow it up'), 
    # because relevant details might also get lost in erosion 
    # (e.g. leaf tips)
    disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)) 
    dilated = cv2.dilate(eroded, disk_kernel)
        
    # sometimes parts of the stem remain (esp. the thicker base) -> remove them
    rem = remove_small_objects(dilated, 10000)
    rem = rem.astype(np.uint8)
    
    # combine more detailed binary with 'blown up' binary   
    rem_resized = cv2.resize(rem, (binary.shape[1], binary.shape[0]))
    masked_binary = mask(binary, rem_resized)    
    masked_seg = mask(segmented_image, masked_binary) 
    return crop(masked_seg)
    
def mask(image, binary):
    binary = np.clip(binary, 0, 1)
    if len(image.shape) == 3:
        masked = image.copy()
        for i in range(masked.shape[2]):
            masked[:, :, i] = np.multiply(masked[:, :, i], binary)
    else:
        masked = image * binary
    return masked

def bounding_box(binary):
    '''
    returns bounding box of given binary image (aroud all pixels != 0)
    (x1, y1, x2, y2) where (x1,y1) is upper left point and (x2, y2) lower right 
    point of bounding box
    '''
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return xmin, ymin, xmax, ymax

def crop(image, border=0):    
    x1, y1, x2, y2 = bounding_box(image)
    x1 = max(x1 - border, 0)
    y1 = max(y1 - border, 0)
    x2 = min(x2 + border, image.shape[1])
    y2 = min(y2 + border, image.shape[0])
    cropped = image[y1: y2, x1: x2]
    # background white rather than black
    image[image==0] = 255
    return cropped

def read_image(filename):    
    pixel_array = cv2.imread(filename, cv2.IMREAD_COLOR) 
    cv2.cvtColor(pixel_array, cv2.COLOR_BGR2RGB, pixel_array)
    return pixel_array        

def write_image(image, filename):
    pixels = image.copy()
    try:
        cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR, pixels)
    except:
        pass
    success = cv2.imwrite(filename, pixels)
    return success