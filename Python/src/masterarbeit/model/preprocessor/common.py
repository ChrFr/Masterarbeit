import numpy as np
    
def mask(image, binary):
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
    (x1, y1, x2, y2) where (x1,y1) is upper left point and (x2, y2) lower right point of bounding box and
    '''
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return xmin, ymin, xmax, ymax

def crop(image):    
    x1, y1, x2, y2 = bounding_box(image)
    cropped = image[y1: y2, x1: x2]
    return cropped