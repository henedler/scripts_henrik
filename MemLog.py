#!/usr/bin/env python
'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import time

import numpy as np
import psutil
from DDFacet.Array import NpShared



def monitorMem():
    t0=time.time()
    Swap0=None


    while True:
        # process = psutil.Process(os.getpid())
        # mem = process.get_memory_info()[0] / float(2 ** 20) 
        vmem=psutil.virtual_memory()
        
        mem=vmem.used/float(2**20)
        memAvail=vmem.available/float(2**20)

        memTotal=vmem.total/float(2**20)

        smem=psutil.swap_memory()
        Smem=smem.used/float(2**20)
        if Swap0 is None:
            Swap0=Smem

        SmemAvail=smem.total/float(2**20)

        # Shared= NpShared.SizeShm()

        cpu=psutil.cpu_percent()

        print(f'Total mem:{memAvail*1e-3:.0f}GB Cache:{memTotal*1e-3:.0f}GB Swap:{Smem*1e-3:.0f}GB')
        time.sleep(10)



class ClassMemMonitor():

    def __init__(self,dt=0.5):
        self.dt=dt
        pass


    def start(self):

        #t = threading.Thread(target=monitorMem)
        #t.start()

        monitorMem()


def test():
    MM=ClassMemMonitor()
    MM.start()
    

if __name__=="__main__":
    MM=ClassMemMonitor()
    MM.start()
