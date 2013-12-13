#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distutils.core import setup

# Scripts to be installed to some dir
scripts = [ "scripts/crisppose", 
            "scripts/crisptime"]

setup(name='crisp',
      version='1.0',
      author="Hannes Ovrén",
      author_email="hannes.ovren@liu.se",
      description="Camera-to-IMU calibration and synchronization toolkit",
      license="GPL",
      scripts=scripts,
      packages=['crisp'],
      )
