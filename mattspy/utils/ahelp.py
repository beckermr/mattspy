"""
Utilities for using and manipulating numerical python arrays (NumPy).

    ahelp(array, recurse=False, pretty=True)
        Print out a formatted description of the input array.   If the array
        has fields, individual descriptions are printed for each field.  This
        is designed to be similar to help, struct, /str in IDL.

"""

license="""
  Copyright (C) 2010  Erin Sheldon

    This program is free software; you can redistribute it and/or modify it
    under the terms of version 2 of the GNU General Public License as
    published by the Free Software Foundation.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""

import sys
from sys import stdout, stderr
import pydoc
import numpy

def ahelp(array_in, recurse=False, pretty=True, index=0, page=False, max_vec_print=10):
    """
    Name:
      ahelp()

    Purpose:
        Print out a formatted description of the input array.   If the array
        has fields, individual descriptions are printed for each field.  This
        is designed to be similar to help, struct, /str in IDL.

    Calling Sequence:
        ahelp(array, recurse=False, pretty=True, page=False)

    Inputs:
        array: A numpy array.

    Optional Inputs:
        recurse: for sub-arrays with fields, print out a full description.
            default is False.
        pretty:  If True, split field descriptions onto multiple lines if
            the name is longer than 15 characters.  Nicer for the eye, but
            harder for a machine to parse.  Also, strings are surrounded
            by quotes 'string'.  Default is True.
        page: If True, run the output through a pager.
        max_vec_print: maximum number of elements to print for vector fields
            default is 10

    Example:
        ahelp(a)
        size: 1147506  nfields: 27  type: records
          run                >i4  1933
          rerun              |S3  '157'
          camcol             >i2  1
          field              >i4  11
          mjd                >i4  51886
          tai                >f8  array[5]
          ra                 >f8  102.905870701
          dec                >f8  -1.05070432844

    Revision History:
        Created: 2010-04-05, Erin Sheldon, BNL
        Modified to print arrays 2015-12-09, Matthew R. Becker, Stanford
    """


    # make sure the data can be viewed as a
    # numpy ndarray.  pyfits in particular is
    # a problem case that we must get a view of
    # as ndarray.
    if not hasattr(array_in,'view'):
        raise ValueError("data must be an array or have the .view method")

    array = array_in.view(numpy.ndarray)

    names = array.dtype.names
    descr = array.dtype.descr


    topformat="size: %s  nfields: %s  type: %s\n"

    lines=[]
    if names is None:
        type=descr[0][1]
        nfields=0
        line=topformat % (array.size, nfields, type)
        lines.append(line)
        #stdout.write(line)

    else:
        line=topformat % (array.size, len(names), 'records')
        lines.append(line)
        #stdout.write(line)
        flines=_get_field_info(array,
                               recurse=recurse,
                               pretty=pretty,
                               index=index,
                               max_array_printlen=max_vec_print)
        lines += flines

    lines='\n'.join(lines)

    if not page:
        stdout.write(lines)
        stdout.write('\n')
    else:
        import pydoc
        pydoc.pager(lines)

def _get_field_info(array, nspace=2, recurse=False, pretty=True, index=0, max_array_printlen=10):
    names = array.dtype.names
    if names is None:
        raise ValueError("array has no fields")

    if len(array.shape) == 0:
        is_scalar=True
    else:
        is_scalar=False


    lines=[]
    spacing = ' '*nspace

    nname = 15
    ntype = 6

    # this format makes something machine readable
    #format = spacing + "%-" + str(nname) + "s %" + str(ntype) + "s  %s\n"
    # this one is prettier since lines wrap after long names
    #pformat = spacing + "%-" + str(nname) + "s\n %" + str(nspace+nname+ntype) + "s  %s\n"


    # this format makes something machine readable
    format = spacing + "%-" + str(nname) + "s %" + str(ntype) + "s  %s"
    # this one is prettier since lines wrap after long names
    pformat = spacing + "%-" + str(nname) + "s\n %" + str(nspace+nname+ntype) + "s  %s"

    max_pretty_slen = 25

    for i in range(len(names)):

        hasfields=False


        n=names[i]

        type=array.dtype.descr[i][1]

        if is_scalar:
            fdata = array[n]
        else:
            fdata = array[n][index]


        if numpy.isscalar(fdata):
            if isinstance(fdata, numpy.string_):
                d=fdata

                # if pretty printing, reduce string lengths
                if pretty and len(d) > max_pretty_slen:
                    d = fdata[0:max_pretty_slen]
                    d = "'" + d +"'"
                    d = d+'...'
                else:
                    if pretty:
                        d = "'" + d +"'"
            else:
                d = fdata
        else:
            shape_str = ','.join( str(s) for s in fdata.shape)
            if fdata.dtype.names is not None:
                type = 'rec[%s]' % shape_str
                d=''
                hasfields=True
            else:
                if len(fdata.shape) < 2 and fdata.size <= max_array_printlen:
                    d = "["
                    for v in fdata.ravel():
                        d += str(v)
                        d += ", "
                    d = d[0:-2]
                    d += "]"
                    #d = fdata.__str__()
                else:
                    d = 'array[%s]' % shape_str

        if pretty and len(n) > 15:
            l = pformat % (n,type,d)
        else:
            l = format % (n,type,d)
        lines.append(l)
        #stdout.write(l)

        if hasfields and recurse:
            #new_nspace = nspace + nname + 1 + ntype + 2
            new_nspace = nspace + 4
            morelines = _get_field_info(array[n],
                                        nspace=new_nspace,
                                        recurse=recurse)
            lines += morelines

    return lines
