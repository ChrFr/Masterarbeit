from PyQt5 import Qt, QtCore, QtGui, QtWidgets
import sys
from UI.main_window_ui import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow
from collections import OrderedDict
import numpy as np
import cv2

from masterarbeit.UI.batch_dialogs import (CropDialog, FeatureDialog,
                                           browse_file)

from masterarbeit.config import (IMAGE_FILTER, ALL_FILES_FILTER)
from masterarbeit.config import PRE_PROCESSORS
from masterarbeit.config import FEATURES
from masterarbeit.model.preprocessor.common import mask, crop, read_image

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
            
        q_image = QtGui.QImage(pixel_array, width, height, byte_value, imformat)        
        self.pixmap = QtGui.QPixmap(q_image)
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


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.source_pixels = None
        self.setup()

    def setup(self):
        
        self.source_view = ImageViewer(self.source_label)
        self.preprocess_view = ImageViewer(self.preprocess_label) 
        
        ### PREPROCESSING VIEW ###        

        # drag and drop images into source view, will be handled by self.eventFilter
        self.source_image_scroll.setAcceptDrops(True)
        self.source_image_scroll.installEventFilter(self)

        # zoom on slider change
        def zoom_changed(factor):
            self.source_view.zoom(factor)
            self.preprocess_view.zoom(factor)        
        self.zoom_slider.valueChanged.connect(zoom_changed)

        # change image when selected from combo containing the preprocessed data
        def image_selected(index):
            if index < 0:
                return
            pixels = self.preprocess_combo.itemData(index)
            self.preprocess_view.draw_pixels(pixels)        
        self.preprocess_combo.currentIndexChanged.connect(image_selected)

        self.preprocess_button.pressed.connect(self.preprocess)      

        self.actionOpen_Image.triggered.connect(
            lambda: self.load_image(
                browse_file(title='Choose Image', 
                            filters=[IMAGE_FILTER, ALL_FILES_FILTER],
                            parent=self)
            )
        )

        ### MENU ###
        
        self.actionCrop_Images.triggered.connect(lambda: CropDialog().exec_())
        self.actionExtract_Features.triggered.connect(
            lambda: FeatureDialog().exec_())
        self.actionExit.triggered.connect(Qt.qApp.quit)      
                
        for processor in PRE_PROCESSORS:
            self.preprocessor_combo.addItem(processor.label, processor)                   

    # called on events connected via installFilterEvent
    def eventFilter(self, object, event):
        if (object is self.source_image_scroll):
            if (event.type() == QtCore.QEvent.DragEnter):
                # dragged item has url (most likely path to file)
                if event.mimeData().hasUrls():
                    event.accept()   
                    return True
                else:
                    event.ignore()
            if (event.type() == QtCore.QEvent.Drop):
                # if dropped item has url -> extract filename 
                if event.mimeData().hasUrls():  
                    filename = event.mimeData().urls()[0].toLocalFile()
                    self.load_image(filename)
                    return True
            return False

    def load_image(self, filename):
        if not filename:
            return
        self.reset_views()
        self.source_pixels = read_image(filename)
        self.source_view.draw_pixels(self.source_pixels)     

    def preprocess(self):
        if self.source_pixels is None:
            return
        self.preprocess_combo.clear()
        steps = OrderedDict()
        index = self.preprocessor_combo.currentIndex()
        preprocessor = self.preprocessor_combo.itemData(index)()
        binary = preprocessor.process(self.source_pixels, steps_dict=steps)
        
        #masked_image = mask(self.source_pixels, binary)
        #cropped = crop(masked_image, border=5)
        #steps['cropped and masked image'] = cropped        
        
        for name, pixels in steps.items():
            self.preprocess_combo.addItem(name, pixels)            
        self.preprocess_combo.setCurrentIndex(self.preprocess_combo.count() - 1)

        #self.descriptor.describe(self.preprocessor.process_steps['binary'])
        #self.preprocess_combo.setCurrentIndex(self.preprocess_combo.count())

    def reset_views(self):
        self.preprocess_combo.clear()
        self.preprocess_label.clear()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()        
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()