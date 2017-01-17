# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QDialog
from PyQt5 import Qt, QtCore, QtGui
import time, datetime
import os, re

from masterarbeit.UI.crop_images_ui import Ui_Dialog
from masterarbeit.UI.progress_ui import Ui_ProgressDialog
from masterarbeit.model.preprocessor.preprocessor_skimage import BinarizeHSV
from masterarbeit.model.preprocessor.common import mask, crop
#from masterarbeit.model.preprocessor.opencv_preprocessor import OpenCVPreProcessor

IMAGE_FILTER = 'Images (*.png, *.jpg)'
ALL_FILES_FILTER = 'All Files(*.*)'

def browse_file(title='Select File', filters=[ALL_FILES_FILTER], multiple=False, parent=None):
    if multiple:
        browse_func = Qt.QFileDialog.getOpenFileNames
    else:
        browse_func = Qt.QFileDialog.getOpenFileName
    filename, filter = browse_func(
            parent, title,
            filter=';;'.join(filters),
            initialFilter=filters[0])
    return filename

def browse_folder(title='Select Folder', parent=None):
    folder = Qt.QFileDialog.getExistingDirectory(
            parent, title)
    return folder


class CropDialog(QDialog, Ui_Dialog):
    def __init__(self):          
        QDialog.__init__(self)
        self.setupUi(self)
        self.setup()
        
    def setup(self):
        # input images
        def set_input_images():
            files = browse_file('Select Input Files', multiple=True, parent=self)
            if len(files) > 0:
                self.input_images_edit.setText(';'.join(files))                
        self.input_images_browse_button.pressed.connect(set_input_images)
        
        # output folder
        def set_output_folder():
            folder = browse_folder('Select Output Folder', parent=self)
            if folder:
                self.input_folder_edit.setText(folder)         
        self.input_folder_browse_button.pressed.connect(set_output_folder) 
                
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
                self.processor = BinarizeHSV
                
            def run(self):
                self.stop_requested = False
                step = 100 / len(input_files)
                progress = 0
                self.status.emit('<b>Start cropping {} files</b><br>'.format(
                    len(input_files)), 0)                
                for input_fp, output_fp in self.in_out:
                    if self.stop_requested:
                        break
                    
                    image = self.processor.read(input_fp)
                    if not os.path.exists(input_fp):
                        text = '<font color="red">{f} skipped, does not exist</font>'.format(f=input_fp)
                    elif re.search('[ü,ä,ö,ß,Ü,Ö,Ä]', input_fp):
                        text = '<font color="red">{f} skipped, Umlaute not supported in OpenCV</font>'.format(f=input_fp)
                    else:
                        binary = self.processor.process(image)
                        masked_image = mask(image, binary)
                        cropped = crop(masked_image)       
                        out_path = os.path.split(output_fp)[0]
                        if not os.path.exists(out_path):
                            os.makedirs(out_path)
                        success = self.processor.write(cropped, output_fp)                        
                        if success:
                            text = '{f} cropped and written to {o}'.format(
                                f=input_fp, o=output_fp)
                        else:
                            text = '<font color="red">could not write to {o}</font>'.format(o=output_fp)
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
        
        diag = ProgressDialog(CropThread(in_out, suffix))
        diag.exec_()
        
class ProgressThread(QtCore.QThread):
    status = QtCore.pyqtSignal(str, int)
    stop_requested = False
    def run(self):
        raise NotImplementedError
    
    def stop(self):
        self.stop_requested = True

class ProgressDialog(QDialog, Ui_ProgressDialog):

    def __init__(self, progress_thread, parent=None):
        super(ProgressDialog, self).__init__(parent=parent)
        self.parent = parent
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
        delta = datetime.datetime.now() - self.start_time
        h, remainder = divmod(delta.seconds, 3600)
        m, s = divmod(remainder, 60)
        timer_text = '{:02d}:{:02d}:{:02d}'.format(h, m, s)
        self.elapsed_time_label.setText(timer_text)
