# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qt/progress.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ProgressDialog(object):
    def setupUi(self, ProgressDialog):
        ProgressDialog.setObjectName("ProgressDialog")
        ProgressDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        ProgressDialog.resize(635, 347)
        ProgressDialog.setMinimumSize(QtCore.QSize(410, 210))
        ProgressDialog.setMaximumSize(QtCore.QSize(10000, 10000))
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(ProgressDialog)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.log_edit = QtWidgets.QTextEdit(ProgressDialog)
        self.log_edit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.log_edit.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.log_edit.setReadOnly(True)
        self.log_edit.setObjectName("log_edit")
        self.verticalLayout_2.addWidget(self.log_edit)
        self.progress_bar = QtWidgets.QProgressBar(ProgressDialog)
        self.progress_bar.setProperty("value", 0)
        self.progress_bar.setObjectName("progress_bar")
        self.verticalLayout_2.addWidget(self.progress_bar)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.elapsed_time_label = QtWidgets.QLabel(ProgressDialog)
        self.elapsed_time_label.setText("")
        self.elapsed_time_label.setObjectName("elapsed_time_label")
        self.horizontalLayout.addWidget(self.elapsed_time_label)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.start_button = QtWidgets.QPushButton(ProgressDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.start_button.sizePolicy().hasHeightForWidth())
        self.start_button.setSizePolicy(sizePolicy)
        self.start_button.setMinimumSize(QtCore.QSize(87, 23))
        self.start_button.setMaximumSize(QtCore.QSize(87, 23))
        self.start_button.setObjectName("start_button")
        self.horizontalLayout.addWidget(self.start_button)
        self.cancel_button = QtWidgets.QPushButton(ProgressDialog)
        self.cancel_button.setMinimumSize(QtCore.QSize(87, 23))
        self.cancel_button.setMaximumSize(QtCore.QSize(87, 23))
        self.cancel_button.setObjectName("cancel_button")
        self.horizontalLayout.addWidget(self.cancel_button)
        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.retranslateUi(ProgressDialog)
        QtCore.QMetaObject.connectSlotsByName(ProgressDialog)

    def retranslateUi(self, ProgressDialog):
        _translate = QtCore.QCoreApplication.translate
        ProgressDialog.setWindowTitle(_translate("ProgressDialog", "Progress"))
        self.start_button.setText(_translate("ProgressDialog", "Start"))
        self.cancel_button.setText(_translate("ProgressDialog", "Abbrechen"))

