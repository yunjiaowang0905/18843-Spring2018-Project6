__author__ = 'Jin Gong'
import numpy as np

def biharmonic_spline_interp2(X, Y, Z, XI, YI): # the third input should be Z
    """
    2D Biharmonic spline interpolation implemented from:

    Sandwell, D. T. (1987), Biharmonic spline interpolation of GEOS-3 and
    SEASAT altimeter data, Geophysical Research Letters, Vol. 2, p. 139 ? 142.

    Python version adapted by Jin Gong.
    """

    params_num = len(locals())
    # Run an example if no input arguments are found
    if params_num != 5:
        print('Running Peaks Example')
        X = np.random.rand(100, 1) * 6 - 3;
        Y = np.random.rand(100, 1) * 6 - 3;
        Z = peaks(X, Y);
        #TODO implement peaks: z =  3*(1-x).^2.*exp(-(x.^2) - (y+1).^2) ...
         #- 10*(x/5 - x.^3 - y.^5).*exp(-x.^2-y.^2) ...
         #- 1/3*exp(-(x+1).^2 - y.^2)
        XI, YI = np.meshgrid(np.arange(-3, 3, 0.25), np.arange(-3, 3, 0.25))  # not sure if it is nessesary to use float


    # TODO: check length of X, Y and Z must be equal, size of XI, YI must be equal

    # Initialize output
    ZI = np.zeros(XI.shape)
    #  Compute GG matrix for GG*m = d inversion problem
    GG = np.zeros(len(Z),len(Z))

    for i in range(0, len(Z)):
        for j in range(0, len(Z)):
            if i != j:
                mgax = np.sqrt(np.square(X[i] - X[j]) + np.square(Y[i] - Y[j]))
                if mgax >= np.exp(-7):
                    gg[i][j] = np.square(magx) * (np.log(magx) - 1)

    #   Compute model "m" where data "d" is equal to "Z"
    m = np.linalg.lstsq(GG, Z)[0] # Left Matrix Division
    #   Find 2D interpolated surface through irregular/regular X, Y grid points
    gg = np.zeros(m.shape)

    for i in range(0, len(ZI)):    # ZI should be row vector, otherwise may cause error
        for k in range(0 , len(Z)):
            magx = np.sqrt(np.square(XI[i] - X[k]))
            if mgax >= np.exp(-7):
                gg[k] = np.square(magx) * (np.log(magx) - 1)
            else:
                gg[k] = np.square(magx) * (-100)
            ZI[i] = np.sum(gg * m)

    #TODO: Plot result if running example or if no output arguments are found

    # Replace infinite or nan number with 0
    for i in range(len(ZI)):
        if np.isnan(ZI[i]) or np.isinf(ZI[i]):
            ZI[i] = 0

    return ZI
