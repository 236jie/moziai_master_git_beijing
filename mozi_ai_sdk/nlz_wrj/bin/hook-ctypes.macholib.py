# -*- coding: utf-8 -*-
from PyInstaller.utils.hooks import copy_metadata
datas = copy_metadata('pickle5')
import scipy.special.cython_special
