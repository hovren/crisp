#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

#from distutils.core import setup
#from distutils.extension import Extension
#from setuptools.command.sdist import sdist as _sdist
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import sys
import codecs

try:
    import numpy as np
except ImportError:
    print("Please install numpy before building this package")
    raise

try:
    #from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: codecs.open(f, encoding='utf-8').read()

# Fast quaternion integration module
file_ext = 'pyx' if USE_CYTHON else 'c'
fastint_sources = ["crisp/fastintegrate/fastintegrate.{}".format(file_ext),]

ext_modules = [Extension("crisp.fastintegrate", fastint_sources, include_dirs=[np.get_include()]),]

if USE_CYTHON:
    ext_modules = cythonize(ext_modules)

classifiers = [
    'Development Status :: 4 - Beta',

    # Indicate who your project is intended for
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',

    # Pick your license as you wish (should match "license" above)
     'License :: OSI Approved :: GNU General Public License (GPL)',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.4'
]

keywords = 'gyroscope gyro camera imu calibration synchronization'

requires = [ 'numpy',
             'scipy',
             'matplotlib'
]


setup(name='crisp',
      version='2.2',
      author="Hannes Ovrén",
      author_email="hannes.ovren@liu.se",
      url="https://github.com/hovren/crisp",
      description="Camera-to-IMU calibration and synchronization toolkit",
      long_description=read_md('README.md'),
      license="GPL",
      packages=['crisp'],
      ext_modules=ext_modules,
#      cmdclass={'build_ext' : build_ext},
      classifiers=classifiers,
      install_requires=requires,
      requires=requires,
      keywords=keywords,
#      cmdclass={'build_ext' : build_ext}
      )
