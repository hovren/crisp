#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# Scripts to be installed to some dir
scripts = [ "scripts/crisppose", 
            "scripts/crisptime"]
            

# Fast quaternion integration module
fastint_sources = ["crisp/fastintegrate/fastint.pyx",]
fastint_module = Extension("crisp.fastintegrate", fastint_sources)

setup(name='crisp',
      version='1.0',
      author="Hannes Ovrén",
      author_email="hannes.ovren@liu.se",
      description="Camera-to-IMU calibration and synchronization toolkit",
      license="GPL",
      scripts=scripts,
      packages=['crisp'],
      ext_modules=[fastint_module],
      cmdclass={'build_ext' : build_ext},
      )
