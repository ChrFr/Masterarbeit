# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import (QDialog, QInputDialog, QListWidgetItem, QCheckBox,
                             QComboBox, QDialogButtonBox, QPushButton,
                             QGridLayout, QSpacerItem, QSizePolicy, QLabel)
from PyQt5 import Qt, QtCore, QtGui
import time, datetime
import os, re
from collections import OrderedDict

from masterarbeit.UI.crop_images_ui import Ui_Dialog as Ui_CropDialog
from masterarbeit.UI.extract_features_ui import Ui_FeatureDialog
from masterarbeit.UI.progress_ui import Ui_ProgressDialog
from masterarbeit.UI.settings_ui import Ui_SettingsDialog
from masterarbeit.model.preprocessor.preprocessor_skimage import BinarizeHSV
from masterarbeit.model.preprocessor.preprocessor_opencv import Binarize
from masterarbeit.model.preprocessor.common import mask, crop, read_image
from masterarbeit.config import (IMAGE_FILTER, ALL_FILES_FILTER, FEATURES, 
                                 HDF5_FILTER, DATA, Config)

config = Config()

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
        self.thread.stop()

    def _show_status(self, text, progress=None):
        self.log_edit.insertHtml(str(text) + '<br>')
        self.log_edit.moveCursor(QtGui.QTextCursor.End)
        if progress:
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
        
        self.start_button.pressed.connect(self.start)
        self.close_button.pressed.connect(self.close)
        
    def start(self):
        
        class CropThread(ProgressThread):
            
            def __init__ (self, in_out, suffix=None):
                super(CropThread, self).__init__()
                self.in_out = in_out
                self.processor = BinarizeHSV()
                
            def run(self):
                self.stop_requested = False
                step = 100 / len(input_files)
                progress = 0
                self.status.emit('<b>Cropping {} files...</b><br>'.format(
                    len(input_files)), 0)                
                for input_fp, output_fp in self.in_out:
                    if self.stop_requested:
                        break
                    
                    image = self.processor.read(input_fp)
                    if not os.path.exists(input_fp):
                        text = '<font color="red"><i>{f}</i> skipped, does not exist</font>'.format(f=input_fp)
                    elif re.search('[ü,ä,ö,ß,Ü,Ö,Ä]', input_fp):
                        text = '<font color="red"><i>{f}</i> skipped, Umlaute not supported in OpenCV</font>'.format(f=input_fp)
                    else:
                        binary = self.processor.process(image)
                        masked_image = mask(image, binary)
                        cropped = crop(masked_image, border=5)       
                        out_path = os.path.split(output_fp)[0]
                        if not os.path.exists(out_path):
                            os.makedirs(out_path)
                        success = self.processor.write(cropped, output_fp)                        
                        if success:
                            text = '<i>{f}</i> cropped and written to <i>{o}</i>'.format(
                                f=input_fp, o=output_fp)
                        else:
                            text = '<font color="red">could not write to <i>{o}</i></font>'.format(o=output_fp)
                    progress += step
                    self.status.emit(text, progress)
                    
            def process_file(input_file, output_file):
                pass
        
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
        
        diag = ProgressDialog(CropThread(in_out, suffix), parent=self)
        diag.exec_()
        
class FeatureDialog(QDialog, Ui_FeatureDialog):
    
    def __init__(self, preselected=[], parent=None):          
        QDialog.__init__(self, parent=parent)
        self.setupUi(self)
        self.setWindowTitle('Extract Features to {}'.format(config.source))
        self.preselected = preselected
        self.data = config.data()
        # actually not a queue, but a dict. 
        self.file_queue = OrderedDict() 
        self.setup() 
        
    def setup(self):
        self.data.open(config.source)  
        
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
        for feature in FEATURES:
            item = QListWidgetItem(self.features_list)
            checkbox = QCheckBox(feature.label)
            if feature in self.preselected:
                checkbox.setChecked(True)
            self.features_list.setItemWidget(item, checkbox)             
        
        self.start_button.pressed.connect(self.start)
        self.close_button.pressed.connect(self.close)
        
    def enqueue_images(self, images):        
        if len(images) > 0:     
            av_species = self.data.get_categories()
            species, result = SelectSpeciesDialog.get_species(av_species, 
                                                              parent=self)
            if not result:
                return
            if species not in self.file_queue:
                self.file_queue[species]= images
            else:
                self.file_queue[species] += images    
        self.image_queue_text.clear()
        for species, images in self.file_queue.items():
            self.image_queue_text.insertHtml('<b>{}</b><br>'.format(species))
            for f in images:
                self.image_queue_text.insertHtml('<i>{}</i><br>'.format(f))         
        
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
        features = []
        for index in range(self.features_list.count()):
            checkbox = self.features_list.itemWidget(self.features_list.item(index))
            if checkbox.isChecked():
                features.append(FEATURES[index])
        if len(self.file_queue) == 0:
            return
        replace = self.replace_check.isChecked()
        diag = ProgressDialog(FeatureThread(self.file_queue, 
                                            self.data, features, 
                                            replace=replace),
                              parent=self)
        diag.exec_() 
        
    def close(self):       
        self.data.close()      
        super(FeatureDialog, self).close()
        
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
        
class FeatureThread(ProgressThread):
    
    def __init__ (self, files, data, feature_types, replace=False):
        super(FeatureThread, self).__init__()
        self.files = files
        self.processor = Binarize()
        self.data = data
        self.feature_types = feature_types
        self.replace = replace
        
    def run(self):
        self.stop_requested = False
        file_count = 0
        for v in self.files.values():
            file_count += len(v)
        step = 100 / file_count
        progress = 0
        self.status.emit('<b>Extracting features from {} '.format(file_count) +
                         'files...</b><br>', 0)
        features = []      
        for species, files in self.files.items():
            if self.stop_requested:
                break              
            for input_file in files:  
                if self.stop_requested:
                    break              
                image = read_image(input_file)
                if not os.path.exists(input_file):
                    text = ('<font color="red"><i>{f}</i> '.format(f=input_file) + 
                            'skipped, does not exist</font>')
                elif re.search('[ü,ä,ö,ß,Ü,Ö,Ä]', input_file):
                    text = ('<font color="red"><i>{f}</i> '.format(f=input_file) + 
                            'skipped, Umlaute not supported</font>')
                else:
                    binary = self.processor.process(image)  
                    for feat_type in self.feature_types:
                        feat = feat_type(species)
                        feat.extract(binary)
                        features.append(feat)
                        text = '{feature} extracted from <i>{file}</i>'.format(
                            feature=feat.label, file=input_file)
                        self.status.emit(text, progress)                                    
                progress += step                    
        self.data.add_features(features, replace=self.replace)
        #self.data.commit()
        text = '{} features'.format(len(features))
        if self.replace:
            text += ' stored, replacing old entries'
        else:
            text += ' appended to store'
        self.status.emit(text, 100)   
