# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'crop_images.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setEnabled(True)
        Dialog.resize(430, 226)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.suffix_edit = QtWidgets.QLineEdit(Dialog)
        self.suffix_edit.setEnabled(False)
        self.suffix_edit.setObjectName("suffix_edit")
        self.gridLayout.addWidget(self.suffix_edit, 5, 1, 1, 4)
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
        self.gridLayout.addLayout(self.horizontalLayout, 7, 2, 1, 4)
        self.images_edit = QtWidgets.QLineEdit(Dialog)
        self.images_edit.setObjectName("images_edit")
        self.gridLayout.addWidget(self.images_edit, 2, 0, 1, 5)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 0, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 0, 1, 1)
        self.images_browse_button = QtWidgets.QPushButton(Dialog)
        self.images_browse_button.setMaximumSize(QtCore.QSize(30, 16777215))
        self.images_browse_button.setObjectName("images_browse_button")
        self.gridLayout.addWidget(self.images_browse_button, 2, 5, 1, 1)
        self.target_edit = QtWidgets.QLineEdit(Dialog)
        self.target_edit.setObjectName("target_edit")
        self.gridLayout.addWidget(self.target_edit, 4, 0, 1, 5)
        self.target_browse_button = QtWidgets.QPushButton(Dialog)
        self.target_browse_button.setMaximumSize(QtCore.QSize(30, 16777215))
        self.target_browse_button.setObjectName("target_browse_button")
        self.gridLayout.addWidget(self.target_browse_button, 4, 5, 1, 1)
        self.suffix_check = QtWidgets.QCheckBox(Dialog)
        self.suffix_check.setObjectName("suffix_check")
        self.gridLayout.addWidget(self.suffix_check, 5, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem2, 6, 2, 1, 1)

        self.retranslateUi(Dialog)
        self.suffix_check.toggled['bool'].connect(self.suffix_edit.setEnabled)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Crop Images"))
        self.suffix_edit.setText(_translate("Dialog", "_cropped"))
        self.start_button.setText(_translate("Dialog", "Start"))
        self.close_button.setText(_translate("Dialog", "Close"))
        self.label.setText(_translate("Dialog", "Image files"))
        self.label_2.setText(_translate("Dialog", "Target directory"))
        self.images_browse_button.setText(_translate("Dialog", "..."))
        self.target_browse_button.setText(_translate("Dialog", "..."))
        self.suffix_check.setText(_translate("Dialog", "File suffix"))

