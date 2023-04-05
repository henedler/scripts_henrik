#!/usr/bin/env python3

import os, sys
import numpy as np
from grid_sensitivity import Noise

n = Noise(freq=54, mode='LBA_ALL', BW=24, nant=52)
print('With IS:', n.get_noise(dec=12,time=15)*1e6,'uJy/b')

n = Noise(freq=54, mode='LBA_ALL', BW=24, nant=38)
print('Without IS:', n.get_noise(dec=12,time=15)*1e6,'uJy/b')

n = Noise(freq=144, mode='HBA_INNER', BW=48, nant=55)
print('HBA:', n.get_noise(dec=12,time=15)*1e6,'uJy/b')
