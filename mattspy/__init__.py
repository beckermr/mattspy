""""
Matt's collection of python utilities.

Starter Code
------------

#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import progressbar
bar = progressbar.ProgressBar(maxval=len(dat),widgets=[progressbar.Bar(marker='|',left='doing work: |',right=''),' ',progressbar.Percentage(),' ',progressbar.AdaptiveETA()])
bar.start()
bar.update(i+1)
bar.finish()

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Modules/Functions
-----------------

rcols(N,colorscale="rainbow") : produce N colors on a color scale

"""


from rcols import *
from pbar import *
from url_walk import url_walk
from stats import *
