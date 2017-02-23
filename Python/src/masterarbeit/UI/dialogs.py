# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import (QDialog, QInputDialog, QListWidgetItem, QCheckBox,
                             QComboBox, QDialogButtonBox, QPushButton,
                             QGridLayout, QSpacerItem, QSizePolicy, QLabel,
                             QVBoxLayout, QMessageBox)
from PyQt5 import Qt, QtCore, QtGui
import time, datetime
import os, re, sys
from collections import OrderedDict
import numpy as np

from masterarbeit.UI.crop_images_ui import Ui_Dialog as Ui_CropDialog
from masterarbeit.UI.batch_features_ui import Ui_FeatureDialog
from masterarbeit.UI.progress_ui import Ui_ProgressDialog
from masterarbeit.UI.settings_ui import Ui_SettingsDialog
from masterarbeit.model.segmentation.segmentation_skimage import BinarizeHSV
from masterarbeit.model.segmentation.segmentation_opencv import Binarize
from masterarbeit.model.segmentation.helpers import (read_image, write_image)
from masterarbeit.model.features.feature import UnsupervisedFeature
from masterarbeit.config import (IMAGE_FILTER, ALL_FILES_FILTER, FEATURES, 
                                 HDF5_FILTER, DATA, Config, SEGMENTATION)

config = Config()

def error_message(text, parent=None):
    msg = QMessageBox(parent=parent)
    msg.setWindowTitle('Error')
    msg.setIcon(QMessageBox.Warning)
    msg.setText(text) 
    msg.exec_()

def browse_file(title='Select File', filters=[ALL_FILES_FILTER], 
                multiple=False, save=False, parent=None):
    if save:
        browse_func = Qt.QFileDialog.getSaveFileName
    elif multiple:
        browse_func = Qt.QFileDialog.getOpenFileNames
    else:
        browse_func = Qt.QFileDialog.getOpenFileName
    filename, filter = browse_func(
            parent, title,
            filter=';;'.join(filters),
            initialFilter=filters[0],
            options=Qt.QFileDialog.DontConfirmOverwrite,
    )
    return filename

def browse_folder(title='Select Folder', parent=None):
    folder = Qt.QFileDialog.getExistingDirectory(
            parent, title)
    return folder

def seconds_to_hms(seconds):            
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return h, m, s

class SettingsDialog(QDialog, Ui_SettingsDialog):    
    def __init__(self, parent=None):
        super(SettingsDialog, self).__init__(parent=parent) 
        self.parent = parent  
        self.setupUi(self)
        for data in DATA:
            self.data_combo.addItem(data.__name__, data)
        data_index = self.data_combo.findText(config.data.__name__)
        self.data_combo.setCurrentIndex(data_index)
        
        # TODO: sql source without browsing      
        self.source_browse_button.setEnabled(True)
        self.source_edit.setReadOnly(True)
        self.source_edit.setText(config.source)
        def set_source():
            source = browse_file(title='Select Source File',
                                save=True, filters=[HDF5_FILTER],
                                parent=self)
            if len(source) > 0:
                if not os.path.exists(source):
                    open(source, 'a').close()
                self.source_edit.setText(source)
        self.source_browse_button.pressed.connect(set_source)
        self.button_box.accepted.connect(self.check_config)
        self.button_box.rejected.connect(self.reject)  
        
    def check_config(self):
        source = self.source_edit.text()
        if len(source) == 0:
            return        
        data = self.data_combo.currentData()
        #new_config = config._config.copy()
        #new_config['source'] = source
        #new_config['data'] =  data
        #shared_items = set(new_config.items()) & set(config.items())
        #if (shared_items)
        config.source = source
        config.data = data
        self.accept()


class ProgressDialog(QDialog, Ui_ProgressDialog):

    def __init__(self, progress_thread, parent=None):
        super(ProgressDialog, self).__init__(parent=parent)
        self.thread = progress_thread
        self.thread.status.connect(self._show_status)
        self.setupUi(self)
        #self.setWindowTitle('')
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.cancel_button.clicked.connect(self.close)
        self.start_button.clicked.connect(self.run)

        # Just to prevent accidentally running multiple times
        # Disable the button when process starts, and enable it when it finishes
        self.thread.started.connect(self._set_running)
        self.thread.finished.connect(self._set_finished)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_timer)
        self.killed = False

        self.start_button.clicked.emit()

    def run(self):
        self.thread.start()

    def _set_running(self):      
        self.killed = False  
        self.start_time = datetime.datetime.now()
        self.timer.start(1000)
        self.progress_bar.setValue(0)
        self.last_update = 0
        self.estimated = (0, 0, 0)
        self.start_button.setEnabled(False)
        self.cancel_button.setText('Stop')
        self.cancel_button.clicked.disconnect(self.close)
        self.cancel_button.clicked.connect(self.kill)

    def _set_stopped(self):
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.start_button.setText('Restart')
        self.cancel_button.setText('Close')
        self.cancel_button.clicked.disconnect(self.kill)
        self.cancel_button.clicked.connect(self.close)

    def _set_finished(self, code=None):
        if not self.killed:
            self.progress_bar.setValue(100)
            self._show_status('<br><b>Done</b><br>')
        else:
            self._show_status('<br><b><font color="red">Aborted</font></b><br>')
        self._set_stopped()

    def kill(self):
        self.killed = True
        self.timer.stop()
        self._show_status(
            '<br>Trying to stop at next possible break point...<br>')
        self.thread.stop()

    def _show_status(self, text, progress=None):
        self.log_edit.insertHtml(str(text) + '<br>')
        self.log_edit.moveCursor(QtGui.QTextCursor.End)
        if progress and progress >= 0:
            self.progress_bar.setValue(progress)

    def _update_timer(self):
        progress = self.progress_bar.value()
        delta = datetime.datetime.now() - self.start_time
        h, m, s = seconds_to_hms(delta.seconds)
        if self.last_update != progress:
            self.last_update = progress
            est_seconds = delta.seconds * 100 / progress
            self.estimated = seconds_to_hms(int(est_seconds))
        est_h, est_m, est_s = self.estimated
        timer_text = '{:02d}:{:02d}:{:02d} (est.: {:02d}:{:02d}:{:02d})'.format(
            h, m, s, est_h, est_m, est_s)
        self.elapsed_time_label.setText(timer_text)


class ProgressThread(QtCore.QThread):
    status = QtCore.pyqtSignal(str, int)
    stop_requested = False
    def run(self):
        raise NotImplementedError

    def stop(self):
        self.stop_requested = True
        
class FunctionProgressDialog(ProgressDialog):
    '''
    generic progress dialog executing the given function and tracking the 
    stdout of the function (no progress indication though)
    '''
    def __init__(self, function, parent=None):
        class WrappingThread(ProgressThread):            
            def __init__ (self, function):
                super(WrappingThread, self).__init__()
                self.function = function                             
            def run(self):
                # redirect stdout to status signal
                emitter = self.status   
                class StdoutEmitter():
                    def write(self, message):
                        emitter.emit(message, None)
                    def flush(self):
                        pass
                sys.stdout = StdoutEmitter()
                self.function()
                self.status.emit('', 100)
        progress_thread = WrappingThread(function)
        super(FunctionProgressDialog, self).__init__(progress_thread, 
                                                     parent=parent)
        # you can't stop the function
        self.cancel_button.setEnabled(False)
                
    def _set_finished(self):
        super(FunctionProgressDialog, self)._set_finished()
        # reset stdout
        sys.stdout = sys.__stdout__
        self.cancel_button.setEnabled(True)
        
        
class CropDialog(QDialog, Ui_CropDialog):
    def __init__(self, parent=None):          
        QDialog.__init__(self, parent=parent)
        self.setupUi(self)
        self.setWindowTitle('Image Segmentation & Crop')
        self.setup()
        
    def setup(self):
        # input images
        def set_input_images():
            files = browse_file('Select Input Files', multiple=True, parent=self)
            if len(files) > 0:
                self.input_images_edit.setText(';'.join(files))                
        self.input_images_browse_button.pressed.connect(set_input_images)
        
        # output folder
        def set_input_folder():
            folder = browse_folder('Select Output Folder', parent=self)
            if folder:
                self.input_folder_edit.setText(folder)         
        self.input_folder_browse_button.pressed.connect(set_input_folder) 
                
        # output folder
        def set_output_folder():
            folder = browse_folder('Select Output Folder', parent=self)
            if folder:
                self.output_folder_edit.setText(folder)                
        self.output_folder_browse_button.pressed.connect(set_output_folder)  
        
        for processor in SEGMENTATION:
            self.segmentation_combo.addItem(processor.label, processor)         
        
        self.start_button.pressed.connect(self.start)
        self.close_button.pressed.connect(self.close)
        
    def start(self):
        
        class CropThread(ProgressThread):
            
            def __init__ (self, in_out, segmentation, suffix=None):
                super(CropThread, self).__init__()
                self.in_out = in_out
                self.segmentation = segmentation()
                
            def run(self):
                self.stop_requested = False
                step = 100 / len(input_files)
                progress = 0
                self.status.emit('<b>Cropping {} files...</b><br>'.format(
                    len(input_files)), 0)                
                for input_fp, output_fp in self.in_out:
                    if self.stop_requested:
                        break
                    
                    image = read_image(input_fp)
                    if not os.path.exists(input_fp):
                        text = '<font color="red"><i>{f}</i> skipped, does not exist</font>'.format(f=input_fp)
                    elif re.search('[ü,ä,ö,ß,Ü,Ö,Ä]', input_fp):
                        text = '<font color="red"><i>{f}</i> skipped, Umlaute not supported in OpenCV</font>'.format(f=input_fp)
                    else:
                        segmented = self.segmentation.process(image)      
                        out_path = os.path.split(output_fp)[0]
                        if not os.path.exists(out_path):
                            os.makedirs(out_path)
                        success = write_image(segmented, output_fp)                        
                        if success:
                            text = '<i>{f}</i> cropped and written to <i>{o}</i>'.format(
                                f=input_fp, o=output_fp)
                        else:
                            text = '<font color="red">could not write to <i>{o}</i></font>'.format(o=output_fp)
                    progress += step
                    self.status.emit(text, progress)
                    
        input_files = []
        output_files = []
        output_folder = self.output_folder_edit.text()  
        if not self.input_folder_check.isChecked():            
            input_files = self.input_images_edit.text().split(';')
        else: 
            input_folder = self.input_folder_edit.text()
            for root, subfolders, files in os.walk(input_folder):
                for file in files:
                    fn, ext = os.path.splitext(file)
                    if ext.lower() == '.jpg':
                        input_files.append(os.path.join(root,file))
        for input_file in input_files:
            path, input_fn = os.path.split(input_file)            
            f, ext = os.path.splitext(input_fn)
            subfolder = ''
            suffix = ''
            if self.suffix_check.isChecked():
                suffix = self.suffix_edit.text() 
            output_fn = f + suffix + ext
            if self.input_folder_check.isChecked():
                subfolder = os.path.relpath(path, input_folder)
            output_fp = os.path.join(output_folder, subfolder, output_fn) 
            output_files.append(output_fp)
        in_out = zip(input_files, output_files)
        
        index = self.segmentation_combo.currentIndex()
        segmentation = self.segmentation_combo.itemData(index)
        
        diag = ProgressDialog(CropThread(in_out, segmentation, suffix=suffix), 
                              parent=self)
        diag.exec_()
        
class BatchFeatureDialog(QDialog, Ui_FeatureDialog):
    
    def __init__(self, preselected=[], parent=None):          
        QDialog.__init__(self, parent=parent)
        self.setupUi(self)
        self.preselected = preselected
        self.store = config.data()
        self.setup() 
        
    def setup(self):
        self.store.open(config.source)  
        
        # drag and drop images into image queue, 
        # will be handled by self.eventFilter
        self.image_queue_text.setAcceptDrops(True)
        self.image_queue_text.installEventFilter(self)        
            
        # input images
        def browse_images():
            files = browse_file('Select Images of ONE species', 
                                filters=[IMAGE_FILTER],
                                multiple=True, parent=self)
            self.enqueue_images(files)
        self.images_browse_button.pressed.connect(browse_images)
        
        # available features
        for feature_type in FEATURES:            
            label = feature_type.label          
            item = QListWidgetItem(self.features_list)
            checkbox = QCheckBox(label)
            if feature_type in self.preselected:
                checkbox.setChecked(True)
            self.features_list.setItemWidget(item, checkbox)             
        
        self.clear_button.pressed.connect(self.clear_queue)
        self.start_button.pressed.connect(self.start)
        self.close_button.pressed.connect(self.close)
        
    def enqueue_images(self, images):        
        raise NotImplementedError
    
    def clear_queue(self):        
        raise NotImplementedError
        
    # called on events connected via installFilterEvent
    def eventFilter(self, object, event):
        if (object is self.image_queue_text):
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
                    filenames = [url.toLocalFile() 
                                 for url in event.mimeData().urls()]
                    self.enqueue_images(filenames)
                    return True
            return False                      
        
    def start(self):         
        pass
        
    def close(self):       
        self.store.close()      
        super(BatchFeatureDialog, self).close()
        
class BuildDictionaryDialog(BatchFeatureDialog):
    def __init__(self, preselected=[], parent=None):
        super(BuildDictionaryDialog, self).__init__(preselected, parent)
        self.setWindowTitle('Extract Dictionary to {}'.format(config.source))
        for index in range(self.features_list.count()):
            feat_type = FEATURES[index]                
            checkbox = self.features_list.itemWidget(
                self.features_list.item(index))            
            if not issubclass(feat_type, UnsupervisedFeature):   
                checkbox.setChecked(False)
                checkbox.setEnabled(False)
        # actually not a queue, but a dict. 
        self.files = []
        
    def enqueue_images(self, images):        
        self.files = images       
        for file in self.files:
            self.image_queue_text.insertHtml('<i>{}</i><br>'.format(file))   
            
    def clear_queue(self):        
        self.files = []
        self.image_queue_text.clear()
        
    def start(self):
        self.processor = Binarize()
        feat_types = []
        for index in range(self.features_list.count()):
            checkbox = self.features_list.itemWidget(
                self.features_list.item(index))
            if checkbox.isChecked():
                feat_types.append(FEATURES[index])
        if len(self.files) == 0 or len(feat_types) == 0:
            return
        store = config.data()        
        store.open(config.source)        
        for feat_type in feat_types:
            build_codebook(feat_type, self.files, store)            
        
class ExtractFeatureDialog(BatchFeatureDialog):
    
    def __init__(self, preselected=[], parent=None):
        super(ExtractFeatureDialog, self).__init__(preselected, parent)
        # actually not a queue, but a dict. 
        self.setWindowTitle('Extract Features to {}'.format(config.source))
        self.file_queue = OrderedDict() 
        self.av_species = self.store.get_categories()
        self.replace_check = QCheckBox('replace existing feature/species ' +
                                       'combinations in store')
        self.replace_check.setChecked(True)
        #self.crop_check = QCheckBox('segment and crop the images, before ' +
                                    #'extracting the features (not needed ' +
                                    #'if already cropped)') 
        
        #self.gridLayout.addWidget(self.crop_check, 9, 0, 1, 3)
        self.codebook_check = QCheckBox('build (and replace) codebook ' +
                                        'on the fly') 
        
        self.checkbox_layout.addWidget(self.replace_check)                      
        self.checkbox_layout.addWidget(self.codebook_check)
        
        self.browse_button = QPushButton('add folder')
        self.button_layout.addWidget(self.browse_button)        
        
        self.browse_button.pressed.connect(self.browse_folder)
        
    def enqueue_images(self, images): 
        if len(images) > 0:     
            species, result = SelectSpeciesDialog.get_species(self.av_species, 
                                                              parent=self)
            if not result:
                return
            if species not in self.file_queue:
                self.file_queue[species]= images
            else:
                self.file_queue[species] += images
        self._update_textbox()
        
    def browse_folder(self):
        def add_folder(folder):
            for lst in os.listdir(folder):
                l_folder = os.path.join(folder, lst)
                if os.path.isdir(l_folder):
                    species = lst
                    if species not in self.file_queue:
                        self.file_queue[species] = []
                    for f in os.listdir(l_folder):
                        f_folder = os.path.join(l_folder, f)
                        if (os.path.isfile(f_folder) and 
                            f.lower().endswith('jpg') or 
                            f.lower().endswith('png')):
                            self.file_queue[species].append(f_folder)
            
        folder = browse_folder('Select Output Folder', parent=self)
        if folder:
            diag = WaitDialog(lambda: add_folder(folder),                           
                              parent=self) 
            diag.finished.connect(self._update_textbox)
            diag.run()        
        
    def _update_textbox(self):
        self.image_queue_text.clear() 
        for species, images in self.file_queue.items():
            self.image_queue_text.insertHtml('<b>{}</b><br>'.format(species))
            for f in images:
                self.image_queue_text.insertHtml('<i>{}</i><br>'.format(f))  
                
    def clear_queue(self):        
        self.file_queue.clear()     
        self.image_queue_text.clear()           
    
    def start(self):
        feature_types = []
        for index in range(self.features_list.count()):
            checkbox = self.features_list.itemWidget(
                self.features_list.item(index))
            if checkbox.isChecked():
                feature_types.append(FEATURES[index])
        if len(self.file_queue) == 0:
            return
        replace = self.replace_check.isChecked()
        build_codebook = self.codebook_check.isChecked()
        diag = ProgressDialog(ExtractFeatureThread(self.file_queue, 
                                                   self.store, feature_types, 
                                                   build_codebook=build_codebook,
                                                   replace=replace),
                              parent=self)
        diag.exec_()     

class SelectSpeciesDialog(QDialog):
    def __init__(self, species, parent=None):
        super(SelectSpeciesDialog, self).__init__(parent=parent)
        self.setWindowTitle('Select Species')
        layout = QGridLayout(self)
        label = QLabel('species shown in pictures:', self)

        self.species_combo = QComboBox(self)
        self.species_combo.setMinimumWidth(300)
        for s in species:
            self.species_combo.addItem(s)
        add_species_button = QPushButton('New', self)
        
        def add_species():
            name, ok = QInputDialog.getText(self, 'Add Species', 
                                            'name of species:')
            if name and ok:         
                # does species already exist?
                index = self.species_combo.findText(name, 
                                                    QtCore.Qt.MatchFixedString)
                if index >= 0:  
                    self.species_combo.setCurrentIndex(index)
                self.species_combo.addItem(name)
                self.species_combo.setCurrentIndex(
                    self.species_combo.count() - 1 )
                
        add_species_button.pressed.connect(add_species)
        spacer = QSpacerItem(20, 10, QSizePolicy.Fixed)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        
        def check_selection():
            if len(self.species_combo.currentText()) > 0:
                self.accept()
        buttons.accepted.connect(check_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(label, 0, 0)
        layout.addWidget(self.species_combo, 1, 0)
        layout.addWidget(add_species_button, 1, 1)
        layout.addItem(spacer, 2, 0)
        layout.addWidget(buttons, 3, 0)        
                    

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def get_species(species, parent = None):
        dialog = SelectSpeciesDialog(species, parent)
        result = dialog.exec_()
        selection = dialog.species_combo.currentText()   
        return selection, result == QDialog.Accepted
    
class SelectionDialog(QDialog):
    def __init__(self, options, label, title, parent):
        super(SelectionDialog, self).__init__(parent=parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        label = QLabel(label, self)

        self.combo = QComboBox(self)
        self.combo.setMinimumWidth(300)
        for opt, data in options:
            self.combo.addItem(opt, data)            
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(label)
        layout.addWidget(self.combo)
        layout.addWidget(buttons)        
                    

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def get_selection(options, label='select', 
                      title='select', parent=None):
        dialog = SelectionDialog(options, label, title, parent)
        result = dialog.exec_()
        selection = dialog.combo.currentText()
        data = dialog.combo.currentData()
        return selection, data, result == QDialog.Accepted
        
class ExtractFeatureThread(ProgressThread):
    
    def __init__ (self, files, data, feature_types, build_codebook=False, 
                  replace=False, store_every_n_step=100):
        super(ExtractFeatureThread, self).__init__()
        self.files = files
        self.processor = Binarize()
        self.store = data
        self.feature_types = feature_types
        self.replace = replace
        self.build_codebook = build_codebook
        self.store_every_n_step = store_every_n_step
        
    def run(self):
        self.stop_requested = False
        file_count = 0
        for v in self.files.values():
            file_count += len(v)
        step = 100 / file_count
        progress = 0
        self.status.emit('<b>Extracting features from {} '.format(file_count) +
                         'files...</b><br>', 0)
        
        feat_codebook_dict = self.get_codebook_dict() if self.build_codebook else {}
        text = ''
        for species, files in self.files.items():
            features_per_species = []   
            if self.stop_requested:
                break              
            for input_file in files:  
                if self.stop_requested:
                    break              
                if not os.path.exists(input_file):
                    text = ('<font color="red"><i>{f}</i> '.format(f=input_file) + 
                            'skipped, does not exist</font>')
                elif re.search('[ü,ä,ö,ß,Ü,Ö,Ä]', input_file):
                    text = ('<font color="red"><i>{f}</i> '.format(f=input_file) + 
                            'skipped, Umlaute not supported</font>')
                else:
                    image = read_image(input_file)
                    binary = None                     
                    for feat_type in self.feature_types:
                        id = os.path.split(input_file)[1]
                        feat = feat_type(species, id=id)
                        if feat_type.binary_input:
                            if binary is None:
                                binary = self.processor.process(image)           
                            input_img = binary
                        else:
                            input_img = image
                        success = feat.describe(input_img)
                        if not success:
                            text = ('failure while extracting {feature} ' +
                                    'from <i>{file}</i>')
                        else:
                            text = '{feature} extracted from <i>{file}</i>'          
                            if feat_type in feat_codebook_dict:
                                codebook = feat_codebook_dict[feat_type]
                                feat.transform(codebook)                                
                            features_per_species.append(feat)                                
                        text = text.format(feature=feat.label, file=input_file)     
                        self.status.emit(text, progress)                                    
                progress += step    
                #if progress % self.store_every_n_step == 0:                    
                    #self.add_features(features)     
                    #features = []
            self.add_features(features_per_species, species)            
        self.status.emit(text, 100)   
        
    def get_codebook_dict(self):
        feat_codebook_dict = {}
        for feat_type in self.feature_types:      
            if issubclass(feat_type, UnsupervisedFeature):
                if self.build_codebook:
                    picked = []
                    # pick 3 images per species to build codebook
                    for species, files in self.files.items():
                        idx = np.random.choice(len(files), 3)
                        picked += list(np.array(files)[idx])
                    self.status.emit(
                        'Building {} '.format(feat_type.codebook_type.__name__) +
                        'from {} randomly picked files'.format(len(picked)) +
                        ' (3 per species)...', -1)
                    feat_codebook_dict[feat_type] = build_codebook(
                        feat_type, picked, self.store)  
                else:
                    feat_codebook_dict[feat_type] = self.store.get_codebook(
                        feat_type)   
        return feat_codebook_dict
        
    def add_features(self, features, species=None):   
        if len(features) == 0:
            return
        self.status.emit('Storing features...', -1)   
        self.store.add_features(features, category=species, replace=self.replace)
        #self.data.commit()
        text = '{} features'.format(len(features))
        if self.replace:
            text += ' stored, replacing old entries'
        else:
            text += ' appended to store'    
        self.status.emit(text, -1)

class WaitDialog(QDialog):
    finished = QtCore.pyqtSignal()
    
    def __init__(self, function, parent=None):
        super(WaitDialog, self).__init__(parent) 
        self.setWindowTitle("...")
        self.thread = self._create_thread(function)
        self.thread.finished.connect(self._close)
        self.done = False
        vbox_layout = QVBoxLayout(self)
        label = QLabel('Please wait...')
        label.setAlignment(QtCore.Qt.AlignCenter)
        vbox_layout.addWidget(label)
        self.setModal(True)
        self.setMinimumSize(200, 75)
        
    def _create_thread(self, function):        
        class WaitThread(QtCore.QThread):   
            finished = QtCore.pyqtSignal()            
            def __init__ (self):
                super(WaitThread, self).__init__()  
                self.result = None
            def run(self):
                try:
                    self.result = function()  
                except Exception as e:
                    self.error = e
                self.finished.emit()
        return WaitThread()
    
    def run(self):   
        self.show()
        self.thread.start()
        
    def _close(self):
        self.done = True
        self.finished.emit()
        self.close()
        
    @property    
    def result(self):
        return self.thread.result
        
    # disable closing of wait window
    def closeEvent(self, evnt):
        if self.done:
            super(WaitDialog, self).closeEvent(evnt)
        else:
            evnt.ignore() 
            
def build_codebook(feat_type, files, store):
    codebook = feat_type.new_codebook()
    processor = Binarize()
    raw_features = []
    print('Extracting features for codebook...')
    for input_file in files:
        #if self.stop_requested:
            #break      
        image = read_image(input_file)
        # category doesn't matter here
        feat = feat_type('')            
        if feat.binary_input:
            image = processor.process(image)  
        success = feat.describe(image)
        if success:
            raw_features.append(feat)
            
    print('Fitting codebook with extracted features')            
    codebook.fit(raw_features)
    store.save_codebook(codebook, feat_type)    
    print('Codebook built and stored.\n')         
    return codebook