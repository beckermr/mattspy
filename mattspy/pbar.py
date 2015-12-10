from __future__ import print_function
import numpy
import time,os,sys
from subprocess import check_output

__all__ = ['PBar']

class PBar(object):
    """
    Prints a progressbar to the screen.
    
    Use like this
    pgr = PBar(N,"doing work")
    pgr.start()
    for i in xrange(N):
        pgr.update(i+1)
    pgr.finish()    
    """
    def __init__(self,Nmax,name=""):
        self.name = name
        if len(name) > 0:
            self.name += ": "
        self.width = None
        self.Nmax = Nmax
        self.di = None
        self.pc = None
        self.ic = 0
        self.lp = None
        self.columnsp = None
        
    def __getdipc(self,slen):
        if slen < self.Nmax:
            self.di = int(float(self.Nmax)/float(slen))
            self.pc = "|"
        else:
            self.di = 1
            self.pc = "|" * int(float(slen)/float(self.Nmax))
        #print slen,self.Nmax,self.di,self.pc

    def _get_width(self):
        try:
            with open(os.devnull, 'w') as silent:
                line = check_output(['stty','size'], stderr=silent)
                columns = line.strip().split()[-1]
        except:
            columns= '80'
        return columns
        
    def start(self):
        self.tstart = time.time()
        self.tp = time.time()
        columns = self._get_width()        
        self.width = int(columns)
        tail = " %3d%% ETA: --:--:--" % 0
        slen = self.width - len(self.name)-len(tail)
        line = self.name + " " * slen + tail
        print(line,end="")
        sys.stdout.flush()
        self.__getdipc(slen)
        self.lp = line
        
    def update(self,i):
        if i-self.ic >= self.di:
            columns = self._get_width()
            
            if self.lp is not None:
                if self.columnsp is not None and self.columnsp > int(columns):
                    sys.stdout.write('\n')
                else:
                    nb = len(self.lp)+1
                    sys.stdout.write('\b' * nb)
                sys.stdout.flush()
            self.width = int(columns)
            dn = int(float(i)/float(self.Nmax)*100.0)
            
            tn = time.time()
            telapsed = tn-self.tstart
            deta = telapsed/float(i)*float(self.Nmax-i)
                        
            dt = tn-self.tp
            eta = dt/float(i-self.ic)*float(self.Nmax-i)
            self.tp = tn
            self.ic = i
            meta = numpy.sqrt(deta*eta)
            
            tail = " %3d%% ETA: " % dn
            tail += time.strftime('%H:%M:%S', time.gmtime(meta))
            tlen = self.width - len(self.name)-len(tail)
            self.__getdipc(tlen)
            clen = int(float(i)/float(self.di))
            if clen > tlen: clen = tlen
            slen = tlen-clen
            line = self.name + self.pc * clen
            if slen > 0:
                line += " " * slen
            line += tail
            print(line,end="")
            sys.stdout.flush()
            self.lp = line
            self.columnsp = int(columns)
            
    def finish(self):
        if self.lp is not None:
            nb = len(self.lp)+1
            sys.stdout.write('\b' * nb)
            sys.stdout.flush()
        columns = self._get_width()
        self.width = int(columns)
        telapsed = time.time()-self.tstart
        tail = " %3d%% Time: " % 100
        tail += time.strftime('%H:%M:%S', time.gmtime(telapsed))
        clen = self.width - len(self.name)-len(tail)
        line = self.name + "|" * clen
        line += tail
        print(line)
        sys.stdout.flush()
