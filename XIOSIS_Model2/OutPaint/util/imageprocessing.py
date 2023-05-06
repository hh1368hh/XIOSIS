import numpy as np

def rescale(img,low,high,Imin=None,Imax=None):

    low=np.array(low)
    high=np.array(high)
    if Imin is not None:
        Imin = np.array(Imin)
    if Imax is not None:
        Imax = np.array(Imax)

    if Imin is None:
        Imin = img.min()
    if Imax is None:
        Imax = img.max()

    out = np.clip(img,Imin,Imax)
    
    out = low + ((out-Imin)/(Imax-Imin))*(high-low)
    
    return out