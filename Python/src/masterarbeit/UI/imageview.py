from PyQt5 import QtGui, QtCore
import numpy as np

class ImageViewer():
    """
    views and rescales images
    """
    def __init__(self, label):    
        """
        Parameters
        ----------
        label : QLabel, the container for drawing the images into
        """   
        self.label = label
        self.size = self.label.size()
        self.current_zoom = 1
        self.pixmap = None

    def draw_pixels(self, pixel_array):
        """
        draw image from given pixels (colored or greyscale)

        Parameters
        ----------
        pixel_array : numpy array, array of pixels, colored: each pixel is described by an array with 3 values for rgb-values, else only one value per pixel
        """
        # expand binary and hsv to rgb color range
        shape = pixel_array.shape
        if pixel_array.max() <= 1:
            pixel_array *= 255
            pixel_array = np.rint(pixel_array)
            
        # only dtype QImage understands (makes strange things if different)
        desired_dtype = np.uint8
        if pixel_array.dtype != np.uint8:
            pixel_array = pixel_array.astype(np.uint8)
            
        height = shape[0] 
        width = shape[1]
        byte_value = 0
        # colored image
        if len(shape) == 3:
            imformat = QtGui.QImage.Format_RGB888
            byte_value = shape[2] * width            
        # greyscale image
        elif len(shape) == 2:
            imformat = QtGui.QImage.Format_Grayscale8  
            
        # QImage-API is missing entries for ndarrays
        # (though it could be handled same way as normal numpy array)
        if type(pixel_array) == np.ndarray:
            pixel_array = np.array(pixel_array)
        self.pixel_array = pixel_array    
        self.q_image = QtGui.QImage(self.pixel_array, width, height, 
                                    byte_value, imformat)        
        self.pixmap = QtGui.QPixmap.fromImage(self.q_image)
        self.zoom()       

    def zoom(self, factor=None):
        """
        scales and redraws the image

        Parameters
        ----------
        size : QSize, width and height the image is scaled to        
        """        
        if factor is not None:
            self.current_zoom = factor
        if self.pixmap is None:
            return
        zoom = QtCore.QSize(self.size.width() * self.current_zoom, self.size.height() * self.current_zoom)
        scaled = self.pixmap.scaled(zoom, QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(scaled)
