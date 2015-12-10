
def rcols(N,cmap=None):
    """
    Produce N colors over a given color scale.
    
    Parameters
    ----------
    N: int
        number of colors
    cmap: a matplotlib.colors.Colormap instance (default: None) 
        if set to None, then the matplotlib rc default is used
    
    Returns
    -------
    cols : list
        list of colors
    """
    import matplotlib.cm
    
    if cmap is None:
        cmap = matplotlib.cm.get_cmap()
        
    if N == 1:
        norm = 1.0
    else:
        norm = float(B-1)
    
    cols = []
    for i in range(N):
        (r,g,b,a) = cmap(float(i)/norm)
        rgb = '#%02x%02x%02x' % (int(r*255),int(g*255),int(b*255))
        cols.append(rgb)

    return cols
