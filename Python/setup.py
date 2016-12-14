# -*- coding: utf-8 -*-
#
from setuptools import setup, find_packages

setup(
    name="masterarbeit",
    version="0.01",
    url='https://github.com/ChrFr/Masterarbeit',
    author='Christoph Franke',
    description="tool for training neural networks",
    classifiers=[
        "Programming Language :: Python",
        "License :: Other/Proprietary License",
        "Natural Language :: English",
        "Operating System :: Windows",
        "Programming Language :: Python",
    ],
    keywords='masterarbeit',
    download_url='',
    license='other',
    packages=find_packages('src', exclude=['default', 'icons', 'ez_setup']),
    namespace_packages=['masterarbeit'],
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,

    install_requires=[
        'setuptools'
    ],

    entry_points={
        'console_scripts': [
            'masterarbeit=main.main:startmain'
        ],
    },
)
