from PyQt5 import Qt, QtCore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDialog, 
                             QListWidgetItem, QCheckBox,
                             QTableWidgetItem, QMessageBox)
import sys
from collections import OrderedDict
import numpy as np
import cv2
import os
from collections import OrderedDict

from masterarbeit.UI.main_window_ui import Ui_MainWindow
from masterarbeit.config import Config
from masterarbeit.config import (IMAGE_FILTER, ALL_FILES_FILTER)
from masterarbeit.config import (SEGMENTATION, FEATURES, CLASSIFIERS)
from masterarbeit.model.features.feature import MultiFeature
from masterarbeit.model.segmentation.common import (mask, crop, read_image,
                                                    remove_thin_objects)
from masterarbeit.UI.imageview import ImageViewer
from masterarbeit.model.features.plot import pairplot
from masterarbeit.UI.dialogs import (CropDialog, ExtractFeatureDialog, 
                                     BuildDictionaryDialog,
                                     SettingsDialog,
                                     browse_file, WaitDialog, 
                                     FunctionProgressDialog)

config = Config()

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.source_pixels = None
        self.preprocessed_pixels = None
        self.store = None
        
        # apply config initially
        self.on_config_changed()
                
        ### MENU ###
        
        def open_image():
            self.load_source_image(
                browse_file(title='Choose Image', 
                            filters=[IMAGE_FILTER, ALL_FILES_FILTER],
                            parent=self)
            )            

        self.actionOpen_Image.triggered.connect(open_image)
        self.open_source_button.pressed.connect(open_image)
        
        self.actionCrop_Images.triggered.connect(
            lambda: CropDialog(parent=self).exec_())
        def multi_extract(features=[]):
            ExtractFeatureDialog(preselected=features, parent=self).exec_()
            self.update_species_table()
        self.actionExtract_Features.triggered.connect(lambda : multi_extract())
        self.add_features_button.pressed.connect(lambda: multi_extract(
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
        
        self.setup_seg_feat_tab()     
        self.setup_training()                

    def setup_seg_feat_tab(self):
        
        self.source_view = ImageViewer(self.source_label)
        self.segmentation_view = ImageViewer(self.segmentation_label) 
        self.feature_view = ImageViewer(self.feature_label)
        
        ### PREPROCESSING VIEW ###        

        # drag and drop images into source view, will be handled by 
        # self.eventFilter
        self.source_image_scroll.setAcceptDrops(True)
        self.source_image_scroll.installEventFilter(self)            

        # zoom on slider change
        def zoom_changed(factor):
            self.source_view.zoom(factor)
            self.segmentation_view.zoom(factor)  
            self.feature_view.zoom(factor)        
        self.zoom_slider.valueChanged.connect(zoom_changed)

        # change image when selected from combo containing the preprocessed data
        def on_image_selected(index, view, combo):
            if index < 0:
                return            
            pixels = combo.itemData(index)
            view.draw_pixels(pixels)      
            
        self.segmentation_combo.addItem('keep source image')                 
        for processor in SEGMENTATION:
            self.segmentation_combo.addItem(processor.label, processor) 
            
        self.segmentation_button.pressed.connect(self.segmentation)        
        self.segmentation_steps_combo.currentIndexChanged.connect(
            lambda index: on_image_selected(index, self.segmentation_view,
                                            self.segmentation_steps_combo))                   
        
        for feature in FEATURES:
            self.feature_combo.addItem(feature.label, feature)  
        self.feature_steps_combo.currentIndexChanged.connect(
            lambda index: on_image_selected(index, self.feature_view,
                                            self.feature_steps_combo))            
        self.extract_feature_button.pressed.connect(self.extract_feature)

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
        self.reset_processed_views()
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
        self.update_feature_table()
        self.color_species_table()
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
            lambda: self.plot_features(self.get_checked_features(),
                              self.get_checked_species(),
                              pairplot)
        )
        
        def mass_select_features(state):      
            for row in range(self.feature_table.rowCount()):
                feature = self.feature_table.item(row, 0)
                # Warning: changing checkstate triggers cellChanged
                # so bound signal is called here every loop 
                # (but no noticable impact on performance by calling 
                #  self.color_species_table multiple times)
                feature.setCheckState(state)  
        self.all_features_check.stateChanged.connect(mass_select_features)
        
        def mass_select_species(state):
            for row in range(self.species_table.rowCount()):
                species_item = self.species_table.item(row, 0)
                species_item.setCheckState(state)
        self.all_species_check.stateChanged.connect(mass_select_species)
        
        for classifier in CLASSIFIERS:
            self.classifier_combo.addItem(classifier.label, classifier)
        self.train_button.pressed.connect(self.train)
        
        def build_dict():
            BuildDictionaryDialog(preselected=self.get_checked_features(), 
                                  parent=self).exec_()
        self.build_dict_button.pressed.connect(build_dict)
            
    def train(self):
        cls = self.classifier_combo.currentData()
        classifier = cls('test')    
        feat_types = self.get_checked_features()
        species = self.get_checked_species()
        feat_type = feat_types[0]
        input_dim = len(feat_type.columns)
        classifier.setup_model(input_dim)
        features = self.store.get_features(feat_type, categories=species)
        
        FunctionProgressDialog(lambda: classifier.train(features), 
                               parent=self).exec_()
        
    def plot_features(self, feature_types, species, plot_func):
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
        from matplotlib import pyplot as plt
        #diag = WaitDialog(lambda: plot_func(features, do_show=False, max_dim=max_dim),                           
                          #parent=self)
        #diag.finished.connect(lambda: plt.show())
        #diag.run() 
        plot_func(features, do_show=True, max_dim=max_dim)
            
    def get_checked_features(self):
        features = []
        table = self.feature_table                
        for row in range(table.rowCount()):
            feature = table.item(row, 0)
            if feature.checkState() > 0:
                features.append(FEATURES[row])
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
            
    def update_feature_table(self):    
        # disconnect signal, else it would be called when adding a row
        try: 
            self.feature_table.cellChanged.disconnect() 
        except: 
            pass        
        
        # available features
        for row, feature in enumerate(FEATURES):
            self.feature_table.insertRow(row)
            feat_item = QTableWidgetItem(feature.label)
            feat_item.setCheckState(False)
            b_color = Qt.QColor(255, 255, 255) 
            if issubclass(feature, MultiFeature):
                dict_label = '{}'.format(feature.dictionary_type.__name__)
                if self.store.get_dictionary(feature) is None:
                    dict_label += ' not built yet'
                    b_color = Qt.QColor(255, 230, 230)
                else:
                    dict_label += ' is built'
                    b_color = Qt.QColor(230, 255, 230)
            else:
                dict_label = 'no dictionary needed'
                       
            dict_item = QTableWidgetItem(dict_label)
            dict_item.setBackground(b_color)
            self.feature_table.setItem(row , 0, feat_item) 
            self.feature_table.setItem(row , 1, dict_item) 
                
        self.feature_table.resizeColumnsToContents()   
        self.feature_table.cellChanged.connect(self.color_species_table)

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
 
    def segmentation(self):
        if self.source_pixels is None:
            return
        self.reset_processed_views()
        
        steps = OrderedDict()        
        def update_ui(result):
            if self.remove_thin_check.isChecked():
                result = remove_thin_objects(result)
            if self.mask_check.isChecked():
                result = mask(self.source_pixels, result)
                result[result == 0] = 255
            steps['result'] = result
            for name, pixels in steps.items():
                self.segmentation_steps_combo.addItem(name, pixels)            
            self.segmentation_steps_combo.setCurrentIndex(
                self.segmentation_steps_combo.count() - 1)
            self.preprocessed_pixels = result
            
        index = self.segmentation_combo.currentIndex()
        # index 0 keeps source pixels
        if index == 0:
            update_ui(self.source_pixels)
        else:
            segmentation = self.segmentation_combo.itemData(index)()
            diag = WaitDialog(lambda: segmentation.process(self.source_pixels, 
                                                           steps=steps),                           
                                      parent=self)
            
            diag.finished.connect(lambda: update_ui(diag.result))
            diag.run()                     
    
    def extract_feature(self):
        if self.preprocessed_pixels is None:
            return
        self.feature_steps_combo.clear()
        steps = OrderedDict()
        index = self.feature_combo.currentIndex()
        feature = self.feature_combo.itemData(index)('-')        
        #diag = WaitDialog(lambda: feature.describe(self.preprocessed_pixels, 
                                                   #steps=steps),                           
                                  #parent=self)
        
        def update_features():
            for name, pixels in steps.items():
                self.feature_steps_combo.addItem(name, pixels)            
            self.feature_steps_combo.setCurrentIndex(
                self.feature_steps_combo.count() - 1)
            self.extracted_feature = feature
            self.feature_ouptut.setText(str(feature.values))
            import matplotlib.pyplot as plt
            #columns = feature.columns
            #x_pos = np.arange(len(columns))
            
            #plt.bar(x_pos, feature.values, align='center')
            #plt.xticks(x_pos, columns)
            #plt.xlabel('Usage')
            #plt.title('Programming language usage')
            fig = plt.figure()
            
            width = .35
            ind = np.arange(len(feature.values))
            plt.bar(ind, feature.values, width=width)
            plt.xticks(ind + width / 2, feature.columns)
            
            fig.autofmt_xdate()            
            plt.show()
        #diag.finished.connect(update_features)
        #diag.run()                
        
        feature.describe(self.preprocessed_pixels, steps=steps) 
        if isinstance(feature, MultiFeature):
            dictionary = self.store.get_dictionary(type(feature))
            feature.build_histogram(dictionary)
        update_features()

    def reset_processed_views(self):
        self.segmentation_steps_combo.clear()
        self.segmentation_label.clear()
        self.feature_steps_combo.clear()
        self.feature_label.clear()
        self.preprocessed_pixels = None
        self.extracted_feature = None

def main():
    app = QApplication(sys.argv)
    window = MainWindow()        
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()