"""
Matt's collection of python utilities.

Starter Code
------------

#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

pgr = PBar(N,"doing work")
pgr.start()
for i in xrange(N):
    pgr.update(i+1)
pgr.finish()    

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""

from pbar import PBar
del pbar
from url_walk import url_walk
import stats
import plotting
import util
import des

