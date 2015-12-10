__all__ = ['rcols']

def rcols(N,colorscale="jet"):
    """
    Produce N colors over a given color scale.
    
    Parameters
    ----------
    N : int
        number of colors
    colorscale : string (Default: "jet")
        matplotlib color map name
    
    Returns
    -------
    cols : list
        list of colors        
    """
    import matplotlib.cm
    cols = []
    cmap = matplotlib.cm.get_cmap(name=colorscale)
    for i in range(N):
        (r,g,b,a) = cmap(i/float(N-1))
        rgb = '#%02x%02x%02x' % (int(r*255),int(g*255),int(b*255))
        cols.append(rgb)
    return cols
