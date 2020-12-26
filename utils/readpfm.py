import re
import numpy as np
import sys
 
# YH: The standard function to read a PFM image file and extract data from it.
def readPFM(file):
    
    # YH: Open file with mode 'read' and 'binary'
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None
    
    # YH: Read a line and remove any white spaces at the end of the string
    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        # YH: Portable FloatMap format was designed as a floating-point image format.
        raise Exception('Not a PFM file.')
        
    # YH: Use 're' the Regular Expression library to extract digital numbers
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        # YH: Python 'map' applies function on items
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    # YH: Next line information is for 'scale' information, 
    #     which ALSO contain the information tells if it is little endian or big endian.
    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

