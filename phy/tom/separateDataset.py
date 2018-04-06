import numpy as np
from scipy.io import loadmat, savemat
# from predict import visualize

def separateDataset(data_all):
    nt, n_lat, n_lon = data_all.shape
    data_train = -1 * np.ones(data_all.shape)
    data_test = -1 * np.ones(data_all.shape)

    for it in xrange(nt):
        data_slice = data_all[it, :, :]
        row, col = np.where(data_slice > 0)

        for ii in xrange(len(row)):
            if ii % 10 == 0:
                data_test[it, row[ii], col[ii]] = data_slice[row[ii], col[ii]]
            else:
                data_train[it, row[ii], col[ii]] = data_slice[row[ii], col[ii]]

        # visualize(data_slice, data_train[it, :, :], data_test[it, :, :], 0, False)

    return data_train, data_test

if __name__ == "__main__":
    # TODO: change these file paths if necessary
    data_all = loadmat('../matdata/data_all.mat')['data']
    
    data_train, data_test = separateDataset(data_all)
    savemat('../matdata/data_train.mat', {'data': data_train})
    savemat('../matdata/data_test.mat', {'data': data_test})