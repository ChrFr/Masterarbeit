# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qt/extract_features.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(533, 282)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.features_box = QtWidgets.QGroupBox(Dialog)
        self.features_box.setObjectName("features_box")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.features_box)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout.addWidget(self.features_box, 6, 0, 1, 2)
        self.add_species_button = QtWidgets.QPushButton(Dialog)
        self.add_species_button.setObjectName("add_species_button")
        self.gridLayout.addWidget(self.add_species_button, 2, 1, 1, 1)
        self.species_combo = QtWidgets.QComboBox(Dialog)
        self.species_combo.setObjectName("species_combo")
        self.gridLayout.addWidget(self.species_combo, 2, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.start_button = QtWidgets.QPushButton(Dialog)
        self.start_button.setMaximumSize(QtCore.QSize(80, 16777215))
        self.start_button.setObjectName("start_button")
        self.horizontalLayout.addWidget(self.start_button)
        self.close_button = QtWidgets.QPushButton(Dialog)
        self.close_button.setMaximumSize(QtCore.QSize(80, 16777215))
        self.close_button.setObjectName("close_button")
        self.horizontalLayout.addWidget(self.close_button)
        self.gridLayout.addLayout(self.horizontalLayout, 8, 0, 1, 2)
        self.crop_check = QtWidgets.QCheckBox(Dialog)
        self.crop_check.setObjectName("crop_check")
        self.gridLayout.addWidget(self.crop_check, 5, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 7, 1, 1, 1)
        self.images_edit = QtWidgets.QLineEdit(Dialog)
        self.images_edit.setObjectName("images_edit")
        self.gridLayout.addWidget(self.images_edit, 4, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 3, 0, 1, 1)
        self.images_browse_button = QtWidgets.QPushButton(Dialog)
        self.images_browse_button.setMaximumSize(QtCore.QSize(30, 16777215))
        self.images_browse_button.setObjectName("images_browse_button")
        self.gridLayout.addWidget(self.images_browse_button, 4, 1, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Extract Features"))
        self.features_box.setTitle(_translate("Dialog", "Features"))
        self.add_species_button.setText(_translate("Dialog", "Neue hinzu"))
        self.start_button.setText(_translate("Dialog", "Start"))
        self.close_button.setText(_translate("Dialog", "Close"))
        self.crop_check.setText(_translate("Dialog", "Crop images first"))
        self.label_2.setText(_translate("Dialog", "Species"))
        self.label.setText(_translate("Dialog", "Images"))
        self.images_browse_button.setText(_translate("Dialog", "..."))

