import numpy as np
from scipy.interpolate import griddata, interp2d, Rbf
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import os

# Fill in missing data in a 2D matrix with 2D Interpolation
def interpolate2d(data, interp_func='gaussian'):
    n_lat, n_lon = data.shape
    grid_x, grid_y = np.mgrid[0:n_lat, 0:n_lon]

    points = np.where(data > 0)
    values = data[points]

    # Interpolation with griddata: 
    # grid_interp = griddata(points, values, (grid_x, grid_y), method='linear')

    # Interpolation with interp2d:
    # f_interp = interp2d(points[0], points[1], values, kind='linear')
    # grid_interp = f_interp(np.arange(n_lat), np.arange(n_lon))

    # Interpolation with Rbf:
    try:
        f_interp = Rbf(points[0], points[1], values, function=interp_func, smooth=0)
        grid_interp = f_interp(grid_x, grid_y)

        grid_interp[np.where(grid_interp < 0)] = 0
    except:
        grid_interp = np.zeros((n_lat, n_lon))
    
    return grid_interp

# Fill in missing data at each timestamp
def interpolateAllData(mat_filename, mat_interp_filename):
    data_all = loadmat(mat_filename)['data_all']
    T, n_lat, n_lon = data_all.shape

    data_interp = []
    for i in range(T):
        data_curr = interpolate2d(data_all[i, :, :])
        data_interp.append(data_curr)

    savemat(mat_interp_filename, {'data_interp_all': data_interp})

if __name__ == "__main__":
    # TODO: change these file paths if necessary
    mat_filename = '../matdata/data_all.mat'
    mat_interp_filename = '../matdata/data_interp_all.mat'

    interpolateAllData(mat_filename, mat_interp_filename)

    # for i in range(5):
    #     data = loadmat(mat_filename)['data_all'][i, :, :]
    #     data_interp = interpolate2d(data)

    #     data[np.where(data<0)] = float('nan')
    #     plt.subplot(121)
    #     plt.imshow(data, vmin=0, vmax=50, interpolation='nearest')
    #     plt.subplot(122)
    #     plt.imshow(data_interp, vmin=0, vmax=50, interpolation='nearest')
    #     # plt.savefig('../matdata/data_interp_{}_{}.jpg'.format(interp_func, i))
        
    #     plt.show()
    #     plt.clf()
