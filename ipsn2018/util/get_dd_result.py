__author__ = 'Jin Gong'
import numpy as np

def get_dd_result(data_tr, n_lat, n_lon):
    """
    get result for data driven method
    return: data_tr_interp: interpolated results
    parameters: data_tr: sum of data_pre, data_udp, data_ver
    """
    data_tr_np = np.asarray(data_tr)
    data_tr_interp = np.zeros((n_lat,n_lon))
    idx = np.nonzero(data_tr_np)
    yp, xp = idx
    z = data_tr_np[idx]
     # not sure why yp -> row, xp -> col
     # non zero element
    if yp:
        x1, y1 = np.meshgrid[1: n_lon + 1, 1: n_lat + 1]  # not sure if it is nessesary to use float
        x1 = x1.flatten(); #[1,1,1,2,2,2,...,n_lon + 1]
        y1 = y1.flatten(); #[1,2,3...,n_lat + 1,1,2,3...]
        output_dd_cur = biharmonic_spline_interp2(xp, yp, z, x1, y1)
        #TODO: modify biharmonic_spline_interp2()
        data_tr_interp = output_dd_cur.reshape((n_lat,n_lon), order = 'F') # the scan order of reshape is differenct between python and matlab
    return data_tr_interp
