from PyQt5 import Qt, QtCore, QtGui, QtWidgets
import sys
from main_window import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow
from opencv_preprocessor import OpenCVPreProcessor
import numpy as np
import cv2

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
        draw image from given pixels
        
        Parameters
        ----------
        pixel_array : numpy array, array of pixels, each pixel is described by an array with 3 values for rgb-values
        """
        shape = pixel_array.shape
        # colored
        if len(shape) == 3:
            height, width, byteValue = shape
            byteValue = byteValue * width
            q_image = QtGui.QImage(pixel_array, width, height, byteValue, QtGui.QImage.Format_RGB888)
        # greyscale
        elif len(shape) == 2:
            height, width = shape
            q_image = QtGui.QImage(pixel_array, width, height, 0, QtGui.QImage.Format_Grayscale8)            
        else:
            return
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
        self.setup()
        
    def setup(self):      
        self.source_view = ImageViewer(self.source_label)
        self.preprocess_view = ImageViewer(self.preprocess_label)         
        
        self.preprocessor = OpenCVPreProcessor()  
        
        def slider_moved(factor):
            self.source_view.zoom(factor)
            self.preprocess_view.zoom(factor)
        
        self.zoom_slider.valueChanged.connect(slider_moved)
        
        def combo_changed(index):
            if index < 0:
                return
            pixels = self.preprocess_combo.itemData(index)
            self.preprocess_view.draw_pixels(pixels)
        
        self.preprocess_combo.currentIndexChanged.connect(combo_changed)
        
        self.preprocess_button.pressed.connect(self.preprocess)
        
        def browse_image():
            filters=['Images (*.png, *.jpg)', 'All Files(*.*)']
            filename, filter = Qt.QFileDialog.getOpenFileName(
                    self, 'Choose Image',
                    filter=';;'.join(filters),
                    initialFilter=filters[0])
            if filename:
                self.load_image(filename)
        
        self.actionOpen_Image.triggered.connect(browse_image)
        self.actionExit.triggered.connect(Qt.qApp.quit)      
        
    def load_image(self, filename):
        self.source_pixels = self.preprocessor.read_file(filename)
        self.source_view.draw_pixels(self.source_pixels)     
        
    def preprocess(self):
        preprocessed_images = self.preprocessor.process()
        self.preprocess_combo.clear()
        for name, pixels in preprocessed_images.items():
            self.preprocess_combo.addItem(name, pixels)
        #self.preprocess_combo.setCurrentIndex(self.preprocess_combo.count())
        
def main():
    app = QApplication(sys.argv)
    window = MainWindow()        
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()