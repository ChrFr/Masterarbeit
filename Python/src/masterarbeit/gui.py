from PyQt5 import Qt, QtCore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDialog, 
                             QListWidgetItem, QCheckBox,
                             QTableWidgetItem, QMessageBox)
import sys
from UI.main_window_ui import Ui_MainWindow
from collections import OrderedDict
import numpy as np
import cv2
import os
from collections import OrderedDict

from masterarbeit.config import Config
from masterarbeit.config import (IMAGE_FILTER, ALL_FILES_FILTER)
from masterarbeit.config import PRE_PROCESSORS
from masterarbeit.config import FEATURES
from masterarbeit.model.preprocessor.common import mask, crop, read_image
from masterarbeit.UI.imageview import ImageViewer
from masterarbeit.model.features.plot import pairplot
from masterarbeit.UI.dialogs import (CropDialog, FeatureDialog, SettingsDialog,
                                  browse_file)

config = Config()

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.source_pixels = None
        self.store = None
                
        ### MENU ###
        
        self.actionCrop_Images.triggered.connect(
            lambda: CropDialog(parent=self).exec_())
        def extract(features=[]):
            FeatureDialog(preselected=features, parent=self).exec_()
            self.update_species_table()
        self.actionExtract_Features.triggered.connect(lambda : extract())
        self.extract_feature_button.pressed.connect(lambda: extract(
            self.get_checked_features()))
        
        ### CONFIGURATION
        
        def configure():
            diag = SettingsDialog(parent=self)
            result = diag.exec_()
            if result == QDialog.Accepted:
                config.write()
                self.on_config_changed()
        self.actionSettings.triggered.connect(configure)           
        self.actionExit.triggered.connect(Qt.qApp.quit)           
        
        ### MAIN TABS ###
        
        self.setup_preprocessing()     
        self.setup_training()
                
        # apply config first time
        self.on_config_changed()

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

    def on_config_changed(self):
        if self.store:
            self.store.close()
        self.store = config.data()
        self.store.open(config.source)
        store_txt = 'Store: {} <i>(change in File/Settings)</i>'.format(
            os.path.split(config.source)[1])
        self.store_label.setText(store_txt)
        self.update_species_table()

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
        
        # available features
        for feature in FEATURES:
            item = QListWidgetItem(self.features_list)
            checkbox = QCheckBox(feature.label)
            checkbox.clicked.connect(self.color_species_table)
            self.features_list.setItemWidget(item, checkbox) 
            
        def delete_features():
            for feature in self.get_checked_features():
                self.store.delete_feature_type(feature)
            self.update_species_table()
        self.delete_features_button.pressed.connect(delete_features)
        
        def delete_species():
            checked_species = self.get_checked_species()
            if len(checked_species) == 0:
                return
            for species in checked_species:
                self.store.delete_category(species)
            self.update_species_table()
        self.delete_species_button.pressed.connect(delete_species)
        
        self.pairplot_button.pressed.connect(
            lambda: self.plot(self.get_checked_features(),
                              self.get_checked_species(),
                              pairplot)
        )
        
        def mass_select_features(state):            
            for index in range(self.features_list.count()):
                checkbox = self.features_list.itemWidget(
                    self.features_list.item(index))
                checkbox.setCheckState(state)
            self.color_species_table()
        self.all_features_check.stateChanged.connect(mass_select_features)
        
        def mass_select_species(state):
            for row in range(self.species_table.rowCount()):
                species_item = self.species_table.item(row, 0)
                species_item.setCheckState(state)
        self.all_species_check.stateChanged.connect(mass_select_species)
        
    def plot(self, feature_types, species, plot_func):
        if len(feature_types) == 0 or len(species) == 0:
            return
        if len(feature_types) > 1:
            msg = QMessageBox(parent=self)
            msg.setWindowTitle('Warning')
            msg.setIcon(QMessageBox.Warning)
            msg.setText('You may only plot one feature type at a time.') 
            msg.setInformativeText('Plot {} now?'.format(
                feature_types[0].label)) 
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            retval = msg.exec_()
            if retval == QMessageBox.Cancel:
                return   
            
        feature_type = feature_types[0]
        max_dim = None
        if len(feature_type.columns) > 7:            
            msg = QMessageBox(parent=self)
            msg.setWindowTitle('Warning')
            msg.setIcon(QMessageBox.Warning)
            msg.setText('The selected feature has a very high dimension!')    
            msg.setInformativeText(
                'Should the plot be split into plots with smaller dimensions?') 
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            retval = msg.exec_()
            if retval == QMessageBox.Yes:
                max_dim = 6   
                
        features = self.store.get_features(feature_type, categories=species)  
        plot_func(features, max_dim=max_dim)
            
    def get_checked_features(self):
        features = []
        for index in range(self.features_list.count()):
            checkbox = self.features_list.itemWidget(
                self.features_list.item(index))
            if checkbox.isChecked():
                features.append(FEATURES[index])
        return features
    
    def get_checked_species(self):
        species = []
        table = self.species_table                
        for row in range(table.rowCount()):
            species_item = table.item(row, 0)
            if species_item.checkState() > 0:
                species.append(species_item.text())
        return species    
            
    def color_species_table(self):   
        features = self.get_checked_features()              
        table = self.species_table                
        for row in range(table.rowCount()):
            feat_item = table.item(row, 1)
            species_item = table.item(row, 0)
            av_features = feat_item.features            
            if len(features) == 0:        
                b_color = Qt.QColor(255, 255, 255)
            # all checked features are available
            elif set(features).issubset(av_features):
                b_color = Qt.QColor(230, 255, 230)
            # not all checked features are available
            else:
                b_color = Qt.QColor(255, 230, 230)
            species_item.setBackground(b_color)
            feat_item.setBackground(b_color) 
 
    def update_species_table(self):        
        self._species_features = []
        table = self.species_table
        # remove all rows
        table.setRowCount(0)
        stored_species = self.store.get_categories()
        for row, species in enumerate(stored_species):
            table.insertRow(row)
            species_check = QTableWidgetItem(species)
            species_check.setCheckState(False)
            feat_txts = []
            feats = []
            feat_missing = False
            for f in FEATURES:
                feat_name = f.__name__
                feat_count = self.store.get_feature_count(species, feat_name)
                feat_txt = '{}: {}'.format(f.label, feat_count)
                feat_txts.append(feat_txt)
                if feat_count > 0:
                    feats.append(f)
            feat_item = QTableWidgetItem(' | '.join(feat_txts))
            # there is no function to add extra data, so just appended
            # available features at object (needed to do coloring without
            # updating table again)
            feat_item.features = feats
            table.setItem(row , 0, species_check)              
            table.setItem(row , 1, feat_item)
        self.color_species_table()
 
    def preprocess(self):
        if self.source_pixels is None:
            return
        self.preprocess_steps_combo.clear()
        steps = OrderedDict()
        index = self.preprocessor_combo.currentIndex()
        preprocessor = self.preprocessor_combo.itemData(index)()
        binary = preprocessor.process(self.source_pixels, steps_dict=steps)
        
        for name, pixels in steps.items():
            self.preprocess_steps_combo.addItem(name, pixels)            
        self.preprocess_steps_combo.setCurrentIndex(
            self.preprocess_steps_combo.count() - 1)

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