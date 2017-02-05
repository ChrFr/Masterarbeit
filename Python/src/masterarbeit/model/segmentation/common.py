import numpy as np
import cv2

from skimage.morphology import erosion, dilation, remove_small_objects   
from skimage.morphology import disk, square

def remove_thin_objects(binary):   
    for i in range(1):
        eroded = erosion(binary, selem=disk(16))
    # dilate with much bigger structuring element ('blow it up'), 
    # because relevant details might also get lost in erosion 
    # (e.g. leaf tips)
    for i in range(1):
        dilated = dilation(eroded, selem=disk(32))
    # sometimes parts of the stem remain (esp. the thicker base) -> remove them
    rem = remove_small_objects(dilated.astype(bool), min_size=10000)
    rem = rem.astype(np.uint8)
    
    # combine more detailed binary with 'blown up' binary
    masked_binary = mask(binary, rem)           
    return masked_binary    
    
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
    # (else problems with later repeated binarization)
    image[image==0] = 255
    return cropped

def read_image(filename):    
    pixel_array = cv2.imread(filename, cv2.IMREAD_COLOR) 
    cv2.cvtColor(pixel_array, cv2.COLOR_BGR2RGB, pixel_array)
    return pixel_array        

def write_image(image, filename):
    pixels = image.copy()
    cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR, pixels)
    success = cv2.imwrite(filename, pixels)
    return success