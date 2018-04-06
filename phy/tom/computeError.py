import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import math

def getErrorRate(orig, pred):
    points = np.where(orig > 0)
    diff = np.absolute(pred[points] - orig[points])
    diff_sum = np.sum(np.sum(diff))
    orig_sum = np.sum(np.sum(orig[points]))
    return diff_sum / orig_sum

def getRootMeanSquaredError(orig, pred):
    points = np.where(orig > 0)
    error = np.sum(np.square(pred[points] - orig[points]))
    return math.sqrt(error / len(points[0])) if len(points[0]) > 0 else 0

def computeError(orig, pred):
    nt, _, _ = pred.shape
    errRate = getErrorRate(orig[0:nt, :, :], pred)
    rmse = getRootMeanSquaredError(orig[0:nt, :, :], pred)
    
    return errRate, rmse

if __name__ == "__main__":
    # TODO: change these file paths if necessary
    mat_filename = '../matdata/data_interp_all.mat'

    data_train = loadmat('../matdata/data_train.mat')['data']
    data_pred = loadmat('../matdata/data_pred.mat')['data']
    data_test = loadmat('../matdata/data_test.mat')['data']

    # plt.subplot(211)
    # plt.imshow(data_train[1759, :, :].transpose(), vmin=0, vmax=100, interpolation='nearest')
    # plt.subplot(212)
    # plt.imshow(data_pred[1759, :, :].transpose(), vmin=0, vmax=100, interpolation='nearest')
    
    # plt.show()

    nt, _, _ = data_pred.shape
    print nt

    data_pred[1800:1850, :, :] = 0
    data_test[1800:1850, :, :] = -1

    print "Error Rate: ", getErrorRate(data_test, data_pred)
    print "RMSE: ", getRootMeanSquaredError(data_test, data_pred)

    tstep = 10
    nstep = nt / tstep
    rmse = np.zeros((nstep,))
    errRates = np.zeros((nstep,))
    x = tstep * np.arange(nstep)

    for i in xrange(nstep):
        tmin = i * tstep
        tmax = tmin + tstep
        errRates[i], rmse[i] = computeError(data_test[tmin:tmax, :, :], data_pred[tmin:tmax, :, :])

        if rmse[i] > 100:
            print tmin, tmax

    plt.plot(x, rmse, 'ro')
    plt.show()

    