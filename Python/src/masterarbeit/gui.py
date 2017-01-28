from PyQt5 import Qt, QtCore, QtWidgets
import sys
from UI.main_window_ui import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow
from collections import OrderedDict
import numpy as np
import cv2

from masterarbeit.config import Config
from masterarbeit.config import (IMAGE_FILTER, ALL_FILES_FILTER)
from masterarbeit.config import PRE_PROCESSORS
from masterarbeit.config import FEATURES
from masterarbeit.model.preprocessor.common import mask, crop, read_image
from masterarbeit.UI.imageview import ImageViewer
from masterarbeit.UI.dialogs import (CropDialog, FeatureDialog, SettingsDialog,
                                  browse_file)

config = Config()

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.source_pixels = None
        
        ### MENU ###
        
        self.actionCrop_Images.triggered.connect(
            lambda: CropDialog(parent=self).exec_())
        self.actionExtract_Features.triggered.connect(
            lambda: FeatureDialog(parent=self).exec_())
        
        ### CONFIGURATION
        
        def configure():
            diag = SettingsDialog(parent=self)
            result = diag.exec_()
            if result == QtWidgets.QDialog.Accepted:
                config.write()
                self.apply_config()
        self.actionSettings.triggered.connect(configure)           
        self.actionExit.triggered.connect(Qt.qApp.quit)   
        
        self.apply_config()
        
        ### MAIN TABS ###
        
        self.setup_preprocessing()     
        self.setup_training()

    def setup_preprocessing(self):
        
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
            pixels = self.preprocess_steps_combo.itemData(index)
            self.preprocess_view.draw_pixels(pixels)        
        self.preprocess_steps_combo.currentIndexChanged.connect(image_selected)

        self.preprocess_button.pressed.connect(self.preprocess) 
        
        def open_image():
            self.load_source_image(
                browse_file(title='Choose Image', 
                            filters=[IMAGE_FILTER, ALL_FILES_FILTER],
                            parent=self)
            )            

        self.actionOpen_Image.triggered.connect(open_image)
        self.open_source_button.pressed.connect(open_image)
                 
        for processor in PRE_PROCESSORS:
            self.preprocessor_combo.addItem(processor.label, processor)                   

    def apply_config(self):
        pass

    def load_source_image(self, filename):
        if not filename:
            return
        self.reset_views()
        self.source_pixels = read_image(filename)
        self.source_view.draw_pixels(self.source_pixels)    
        
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
                    self.load_source_image(filename)
                    return True
            return False        
        
    def setup_training(self):
        pass
 
    def update_feature_list(self):
        pass

    def preprocess(self):
        if self.source_pixels is None:
            return
        self.preprocess_steps_combo.clear()
        steps = OrderedDict()
        index = self.preprocessor_combo.currentIndex()
        preprocessor = self.preprocessor_combo.itemData(index)()
        binary = preprocessor.process(self.source_pixels, steps_dict=steps)
        
        #masked_image = mask(self.source_pixels, binary)
        #cropped = crop(masked_image, border=5)
        #steps['cropped and masked image'] = cropped        
        
        for name, pixels in steps.items():
            self.preprocess_steps_combo.addItem(name, pixels)            
        self.preprocess_steps_combo.setCurrentIndex(
            self.preprocess_steps_combo.count() - 1)

        #self.descriptor.describe(self.preprocessor.process_steps['binary'])
        #self.preprocess_combo.setCurrentIndex(self.preprocess_combo.count())

    def reset_views(self):
        self.preprocess_steps_combo.clear()
        self.preprocess_label.clear()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()        
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()