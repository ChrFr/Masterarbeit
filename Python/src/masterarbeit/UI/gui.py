'''
contains the main control of the ui

(c) 2017, Christoph Franke

this file is part of the master thesis 
"Computergestuetzte Identifikation von Pflanzen anhand ihrer Blattmerkmale"
'''
__author__ = "Christoph Franke"

from PyQt5 import Qt, QtCore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDialog, 
                             QListWidgetItem, QCheckBox,
                             QTableWidgetItem, QInputDialog, QMessageBox)
import sys
from collections import OrderedDict
import numpy as np
import cv2
import os
import json
from collections import OrderedDict
import matplotlib
from matplotlib import pyplot as plt

from masterarbeit.UI.main_window_ui import Ui_MainWindow
from masterarbeit.config import Config
from masterarbeit.config import (IMAGE_FILTER, ALL_FILES_FILTER)
from masterarbeit.config import (SEGMENTATION, FEATURES, CLASSIFIERS,
                                 CODEBOOKS)
from masterarbeit.model.segmentation.helpers import (read_image, 
                                                     remove_thin_objects)
from masterarbeit.UI.imageview import ImageViewer
from masterarbeit.model.features.plot import pairplot
from masterarbeit.UI.dialogs import (CropDialog, ExtractFeatureDialog, 
                                     BuildDictionaryDialog,
                                     SettingsDialog, SelectionDialog,
                                     browse_file, WaitDialog, 
                                     FunctionProgressDialog, error_message)

from masterarbeit.model.segmentation.segmentation import KMeansHSVBinarize
from masterarbeit.model.features.feature import UnorderedFeature

config = Config()

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.source_pixels = None
        self.preprocessed_pixels = None
        self.store = None
        self.pred_classifier = None
        
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
        self.actionExtract_Features.triggered.connect(
            lambda: self.multi_extract())
        
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
        self.setup_training_tab()       
        self.setup_prediction_tab()

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
        
        for feature_type in FEATURES:
            label = feature_type.label
            self.feature_combo.addItem(label, feature_type)  
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
        self.update_feature_table()
        self.update_species_table()
        self.update_trained_classifiers()

    def load_source_image(self, filename):
        if not filename or not os.path.isfile(filename):
            return
        self.reset_processed_views()
        self.source_pixels = read_image(filename)
        self.source_view.draw_pixels(self.source_pixels)    
        
    # called on events connected via installFilterEvent
    def eventFilter(self, obj, event):
        # dragged items have url (most likely path to file)
        if event.type() == QtCore.QEvent.DragEnter:            
            if (event.mimeData().hasUrls and
                obj in [self.source_image_scroll, self.prediction_table]):
                    event.accept()   
                    return True
            else:
                event.ignore()
        elif event.type() == QtCore.QEvent.Drop:
            if not event.mimeData().hasUrls():
                event.ignore()                
            elif obj is self.source_image_scroll:
                # extract filename  
                filename = event.mimeData().urls()[0].toLocalFile()
                self.load_source_image(filename)
                return True
            elif obj is self.prediction_table:    
                filenames = [url.toLocalFile() for url in 
                             event.mimeData().urls()]
                self.add_prediction_images(filenames)
        return False        
        
    def setup_training_tab(self):     
        self.update_feature_table()
        self.color_species_table()
        def delete_features():
            for feature in self.get_checked_features():
                self.store.delete_feature_type(feature)
            self.update_feature_table()
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
            self.classifier_training_combo.addItem(classifier.label, classifier)
        self.train_button.pressed.connect(self.train)
        
        self.add_features_button.pressed.connect(lambda: self.multi_extract(
            self.get_checked_features()))              
        
        for codebook in CODEBOOKS:
            self.codebook_combo.addItem(codebook.__name__, codebook)
        self.build_dict_button.pressed.connect(self.build_codebook)        
        
        self.train_button.pressed.connect(self.train)                
        
    def setup_prediction_tab(self):
        self.update_trained_classifiers()
                
        def change_classifier(idx):
            if self.classifier_prediction_combo.currentIndex() < 0:
                return
            cls_type, name = self.classifier_prediction_combo.currentData()
            self.pred_classifier = self.store.get_classifier(cls_type, name)            
            status = json.dumps(self.pred_classifier.meta, indent=4)            
            self.classifier_status.setText(status)            
        self.classifier_prediction_combo.setCurrentIndex(-1)
        self.classifier_prediction_combo.currentIndexChanged.connect(
            change_classifier)
        
        self.prediction_table.setAcceptDrops(True)
        self.prediction_table.installEventFilter(self)
        
        self.predict_button.pressed.connect(self.predict)
        
        self.clear_prediction_button.pressed.connect(
            lambda: self.prediction_table.setRowCount(0))
        
        def remove_classifier():
            cls_type, name = self.classifier_prediction_combo.currentData()
            self.classifier_prediction_combo.setCurrentIndex(-1)
            self.store.delete_classifier(cls_type, name)
            self.update_trained_classifiers()
        self.remove_classifier_button.pressed.connect(remove_classifier)
        
    def predict(self):          
        if self.pred_classifier is None:
            return
        table = self.prediction_table
        if table.rowCount() == 0:
            return
        files = []
        for row in range(table.rowCount()):
            files.append(table.item(row, 0).text())
        segmentation = config.default_segmentation()
        features = []
        feats_used = self.pred_classifier.trained_features
        # load the required codebooks as used while training
        codebook_dict = {}
        for feat_type, codebook_type in feats_used.items():
            if codebook_type is not None:
                codebook_dict[feat_type] = self.store.get_codebook(
                    feat_type, codebook_type)
        for file in files:
            image = read_image(file)
            binary = segmentation.process(image)
            for feat_type in feats_used.keys():
                feature = feat_type('')
                success = feature.describe(binary)
                if not success:
                    print('failure while extracting features from {}'.format(file))
                elif isinstance(feature, UnorderedFeature):
                    codebook = codebook_dict[feat_type]
                    feature.transform(codebook)
                features.append(feature)
        predictions = self.pred_classifier.predict(features)
        for row, pred in enumerate(predictions):
            self.prediction_table.setItem(row , 1, QTableWidgetItem(pred))         
        
    def update_trained_classifiers(self):
        self.classifier_prediction_combo.clear()
        av_classifiers = self.store.list_classifiers()
        for cls_name, names in av_classifiers.items():
            cls_type = None
            for c in CLASSIFIERS:
                if c.__name__ == cls_name:
                    cls_type = c
                    break
            for name in names:
                label = '{} - {}'.format(cls_name, name)
                self.classifier_prediction_combo.addItem(label, (cls_type, name))        
        
    def add_prediction_images(self, images):
        row_count = self.prediction_table.rowCount()
        for i, image in enumerate(images):
            row = i + row_count
            self.prediction_table.insertRow(row)
            img_item = QTableWidgetItem(image)
            self.prediction_table.setItem(row , 0, img_item) 
            
    def build_codebook(self):
        if self.dict_images_radio.isChecked():
            BuildDictionaryDialog(preselected=self.get_checked_features(), 
                                  parent=self).exec_()
        # build dict from raw features
        elif self.dict_features_radio.isChecked():        
            
            feat_types = self.get_checked_features()
            species = self.get_checked_species()
            codebook_type = self.codebook_combo.currentData()     
            diag = None
            
            def build_from_store():
                for feat_type in feat_types:
                    codebook = codebook_type(feat_type)
                    # pick only a few raw features as while building on the fly,
                    # else may run out of memory (raw features tend to be huge)
                    features = self.store.get_features(
                        feat_type, categories=species,
                        samples_per_category=3)
                    if features is None:
                        print('{} not described yet!'.format(feat_type.label))        
                        continue
                    print('building {} for {} from {} sampled features '
                          .format(codebook_type.__name__, feat_type.label, 
                                  len(features)) +
                          '(3 per species)...')
                    codebook.fit(features)
                    self.store.save_codebook(codebook, feat_type)
                    print('codebook built and stored')
                    
            diag = FunctionProgressDialog(build_from_store, parent=self)
            diag.exec_()
        self.update_feature_table()
            
    def train(self):
        feat_types = self.get_checked_features()
        species = self.get_checked_species()
        if len(species) == 0 or len(feat_types) == 0:
            return        
         
        feat_type = feat_types[0]
        if len(feat_types) == 0 or len(species) == 0:
            return
        features = self.store.get_features(feat_type, categories=species)
        ready_features = []
        raw_features = []
        for feature in features:
            if (isinstance(feature, UnorderedFeature) and
                not feature.is_transformed):
                raw_features.append(feature)
            else:
                ready_features.append(feature)         
                
        feat_codebook_dict = {}
        codebook_type = self.codebook_combo.currentData()  
            
        name, ok = QInputDialog.getText(self, 'Classifier', 
                                    'set name of classifier')
        if not name or not ok:         
            return
        cls = self.classifier_training_combo.currentData()        
        classifier = cls(name)   
        
        def train_and_save():
            if len(raw_features) > 0:
                codebooks = {}
                print('transforming raw features with {}'.format(
                    codebook_type.__name__))
                for feature in raw_features:
                    feat_type = type(feature)
                    if feat_type in codebooks:
                        codebook = codebooks[feat_type]
                    else:
                        codebook = self.store.get_codebook(feat_type, 
                                                           codebook_type)
                        codebooks[feat_type] = codebook
                    feature.transform(codebook)
                    ready_features.append(feature)
            print('training classifier...')
            classifier.train(features)
            print('classifier trained')
            self.store.save_classifier(classifier)
            
        FunctionProgressDialog(train_and_save, parent=self).exec_()
        self.update_trained_classifiers()
        
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
        #if len(feature_type.columns) > 7:            
            #msg = QMessageBox(parent=self)
            #msg.setWindowTitle('Warning')
            #msg.setIcon(QMessageBox.Warning)
            #msg.setText('The selected feature has a very high dimension!')    
            #msg.setInformativeText(
                #'Should the plot be split into plots with smaller dimensions?') 
            #msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            #retval = msg.exec_()
            #if retval == QMessageBox.Yes:
                #max_dim = 6   
                                
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
        self.feature_table.setRowCount(0)
        # disconnect signal, else it would be called when adding a row
        try: 
            self.feature_table.cellChanged.disconnect() 
        except: 
            pass        
        
        # available features
        for row, feature_type in enumerate(FEATURES):
            self.feature_table.insertRow(row)
            feat_item = QTableWidgetItem(feature_type.label)
            feat_item.setCheckState(False)
            b_color = Qt.QColor(255, 255, 255) 
            if issubclass(feature_type, UnorderedFeature):
                codebooks = self.store.list_codebooks(feature_type)
                if len(codebooks) == 0:                    
                    dict_label = 'dictionary required, but none found'
                    b_color = Qt.QColor(255, 230, 230)
                else:
                    dict_label = ' | '.join(codebooks)                    
                    b_color = Qt.QColor(230, 255, 230)      
            else:
                dict_label = 'no dictionary needed'
                       
            dict_item = QTableWidgetItem(dict_label)
            dict_item.setBackground(b_color)
            self.feature_table.setItem(row, 0, feat_item) 
            self.feature_table.setItem(row, 1, dict_item) 
                
        self.feature_table.resizeColumnsToContents()   
        self.feature_table.cellChanged.connect(self.color_species_table)

    def update_species_table(self):        
        self._species_features = []
        table = self.species_table
        # remove all rows
        table.setRowCount(0)
        stored_species = self.store.list_categories()
        for row, species in enumerate(stored_species):
            table.insertRow(row)
            species_check = QTableWidgetItem(species)
            species_check.setCheckState(False)
            feat_txts = []
            feats = []
            feat_missing = False
            for feature_type in FEATURES:
                feat_count = self.store.get_feature_count(species, feature_type)
                label = feature_type.label
                feat_txt = '{}: {}'.format(label, feat_count)
                feat_txts.append(feat_txt)
                if feat_count > 0:
                    feats.append(feature_type)
            feat_item = QTableWidgetItem(' | '.join(feat_txts))
            # there is no function to add extra data, so just appended
            # available features to object (not a good style but needed to do 
            # coloring without updating table again)
            feat_item.features = feats
            table.setItem(row , 0, species_check)              
            table.setItem(row , 1, feat_item)
        self.color_species_table()
        self.species_table.resizeColumnsToContents()   
 
    def segmentation(self):
        if self.source_pixels is None:
            return
        self.reset_processed_views()
        
        steps = OrderedDict()        
        def update_ui(result):
            if self.remove_thin_check.isChecked():
                result = remove_thin_objects(result)
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
            result = segmentation.process(self.source_pixels, steps=steps)    
            update_ui(result)
            #diag = WaitDialog(lambda: segmentation.process(self.source_pixels, 
                                                           #steps=steps),                           
                                      #parent=self)
            
            #diag.finished.connect(lambda: update_ui(diag.result))
            #diag.run()                       
            
    def multi_extract(self, features=[]):
        ExtractFeatureDialog(preselected=features, parent=self).exec_()
        self.update_feature_table()
        self.update_species_table()    
    
    def extract_feature(self):
        if self.preprocessed_pixels is None:
            return
        self.feature_steps_combo.clear()
        steps = OrderedDict()
        index = self.feature_combo.currentIndex()
        feature = self.feature_combo.itemData(index)('-')   
        
        def update_features():           
            self.feature_steps_combo.setCurrentIndex(
                self.feature_steps_combo.count() - 1)
            self.extracted_feature = feature
            self.feature_ouptut.setText(str(feature.values))
            
            for name, pixels in steps.items():
                self.feature_steps_combo.addItem(name, pixels) 
    
        def describe():
            feature.describe(self.preprocessed_pixels, steps=steps) 
            if isinstance(feature, UnorderedFeature):
                codebook = self.store.get_codebook(type(feature), None)
                if codebook is None:
                    error_message('Codebook needed but not built yet!', 
                                  parent=self)
                    return False
                else:
                    feature.transform(codebook)                    
            return True        
                
        def plot():
            fig = plt.figure(dpi=200)            
            width = .35
            ind = np.arange(len(feature.values))                
            plt.bar(ind, feature.values, width=width)
            columns = feature.columns
            if columns is None:
                columns = np.arange(0, len(feature.values))
            plt.xticks(ind + width / 2, columns)
            fig.autofmt_xdate()                     
            fig.canvas.draw()
            plot_str = fig.canvas.tostring_rgb()
            data = np.fromstring(plot_str, dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)) 
            steps['values'] = data              
                
        success = describe()
        if success:
            plot()
        update_features() 
        
        # can't make WaitDialog here, cause catching exceptions (for missing 
        # codebook) crashes the thread
        #diag = WaitDialog(describe)            
        #diag.finished.connect(plot_and_update)
        #diag.run()  

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