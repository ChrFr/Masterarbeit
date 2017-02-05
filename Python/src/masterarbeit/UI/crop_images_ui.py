# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qt/crop_images.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setEnabled(True)
        Dialog.resize(473, 309)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.suffix_edit = QtWidgets.QLineEdit(Dialog)
        self.suffix_edit.setEnabled(False)
        self.suffix_edit.setObjectName("suffix_edit")
        self.gridLayout.addWidget(self.suffix_edit, 10, 2, 1, 4)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 12, 3, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 10, 0, 1, 1)
        self.suffix_check = QtWidgets.QCheckBox(Dialog)
        self.suffix_check.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.suffix_check.setObjectName("suffix_check")
        self.gridLayout.addWidget(self.suffix_check, 10, 1, 1, 1)
        self.input_folder_edit = QtWidgets.QLineEdit(Dialog)
        self.input_folder_edit.setEnabled(False)
        self.input_folder_edit.setObjectName("input_folder_edit")
        self.gridLayout.addWidget(self.input_folder_edit, 6, 0, 1, 6)
        self.input_folder_check = QtWidgets.QCheckBox(Dialog)
        self.input_folder_check.setObjectName("input_folder_check")
        self.gridLayout.addWidget(self.input_folder_check, 5, 0, 1, 2)
        self.output_folder_browse_button = QtWidgets.QPushButton(Dialog)
        self.output_folder_browse_button.setMaximumSize(QtCore.QSize(30, 16777215))
        self.output_folder_browse_button.setObjectName("output_folder_browse_button")
        self.gridLayout.addWidget(self.output_folder_browse_button, 9, 6, 1, 1)
        self.input_images_edit = QtWidgets.QLineEdit(Dialog)
        self.input_images_edit.setObjectName("input_images_edit")
        self.gridLayout.addWidget(self.input_images_edit, 4, 0, 1, 6)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 8, 0, 1, 2)
        self.input_images_browse_button = QtWidgets.QPushButton(Dialog)
        self.input_images_browse_button.setMaximumSize(QtCore.QSize(30, 16777215))
        self.input_images_browse_button.setObjectName("input_images_browse_button")
        self.gridLayout.addWidget(self.input_images_browse_button, 4, 6, 1, 1)
        self.output_folder_edit = QtWidgets.QLineEdit(Dialog)
        self.output_folder_edit.setObjectName("output_folder_edit")
        self.gridLayout.addWidget(self.output_folder_edit, 9, 0, 1, 6)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem2, 0, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.start_button = QtWidgets.QPushButton(Dialog)
        self.start_button.setMaximumSize(QtCore.QSize(80, 16777215))
        self.start_button.setObjectName("start_button")
        self.horizontalLayout.addWidget(self.start_button)
        self.close_button = QtWidgets.QPushButton(Dialog)
        self.close_button.setMaximumSize(QtCore.QSize(80, 16777215))
        self.close_button.setObjectName("close_button")
        self.horizontalLayout.addWidget(self.close_button)
        self.gridLayout.addLayout(self.horizontalLayout, 13, 3, 1, 4)
        spacerItem4 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem4, 7, 0, 1, 1)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 3, 0, 1, 5)
        self.input_folder_browse_button = QtWidgets.QPushButton(Dialog)
        self.input_folder_browse_button.setEnabled(False)
        self.input_folder_browse_button.setMaximumSize(QtCore.QSize(30, 16777215))
        self.input_folder_browse_button.setObjectName("input_folder_browse_button")
        self.gridLayout.addWidget(self.input_folder_browse_button, 6, 6, 1, 1)
        self.segmentation_combo = QtWidgets.QComboBox(Dialog)
        self.segmentation_combo.setObjectName("segmentation_combo")
        self.gridLayout.addWidget(self.segmentation_combo, 2, 0, 1, 6)
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.suffix_check.toggled['bool'].connect(self.suffix_edit.setEnabled)
        self.input_folder_check.toggled['bool'].connect(self.input_images_edit.setDisabled)
        self.input_folder_check.toggled['bool'].connect(self.input_folder_edit.setEnabled)
        self.input_folder_check.toggled['bool'].connect(self.input_images_browse_button.setDisabled)
        self.input_folder_check.toggled['bool'].connect(self.input_folder_browse_button.setEnabled)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Crop Images"))
        self.suffix_edit.setText(_translate("Dialog", "_cropped"))
        self.suffix_check.setText(_translate("Dialog", "add file suffix"))
        self.input_folder_check.setText(_translate("Dialog", "Input Folder (all images in folder and subfolders)"))
        self.output_folder_browse_button.setText(_translate("Dialog", "..."))
        self.label_2.setText(_translate("Dialog", "Output Folder"))
        self.input_images_browse_button.setText(_translate("Dialog", "..."))
        self.start_button.setText(_translate("Dialog", "Start"))
        self.close_button.setText(_translate("Dialog", "Close"))
        self.label.setText(_translate("Dialog", "Input Images (manual selection)"))
        self.input_folder_browse_button.setText(_translate("Dialog", "..."))
        self.label_3.setText(_translate("Dialog", "Segmentation"))

