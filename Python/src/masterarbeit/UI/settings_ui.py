# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qt/settings.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SettingsDialog(object):
    def setupUi(self, SettingsDialog):
        SettingsDialog.setObjectName("SettingsDialog")
        SettingsDialog.resize(417, 179)
        self.gridLayout = QtWidgets.QGridLayout(SettingsDialog)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(SettingsDialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.source_edit = QtWidgets.QLineEdit(SettingsDialog)
        self.source_edit.setObjectName("source_edit")
        self.gridLayout.addWidget(self.source_edit, 3, 0, 1, 1)
        self.source_browse_button = QtWidgets.QPushButton(SettingsDialog)
        self.source_browse_button.setEnabled(False)
        self.source_browse_button.setMaximumSize(QtCore.QSize(30, 16777215))
        self.source_browse_button.setObjectName("source_browse_button")
        self.gridLayout.addWidget(self.source_browse_button, 3, 1, 1, 1)
        self.label = QtWidgets.QLabel(SettingsDialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.data_combo = QtWidgets.QComboBox(SettingsDialog)
        self.data_combo.setObjectName("data_combo")
        self.gridLayout.addWidget(self.data_combo, 1, 0, 1, 3)
        self.button_box = QtWidgets.QDialogButtonBox(SettingsDialog)
        self.button_box.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.button_box.setOrientation(QtCore.Qt.Horizontal)
        self.button_box.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.button_box.setObjectName("button_box")
        self.gridLayout.addWidget(self.button_box, 5, 0, 1, 3)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 4, 0, 1, 3)

        self.retranslateUi(SettingsDialog)
        self.button_box.accepted.connect(SettingsDialog.accept)
        self.button_box.rejected.connect(SettingsDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(SettingsDialog)

    def retranslateUi(self, SettingsDialog):
        _translate = QtCore.QCoreApplication.translate
        SettingsDialog.setWindowTitle(_translate("SettingsDialog", "Settings"))
        self.label_2.setText(_translate("SettingsDialog", "Source"))
        self.source_browse_button.setText(_translate("SettingsDialog", "..."))
        self.label.setText(_translate("SettingsDialog", "Datastore"))

