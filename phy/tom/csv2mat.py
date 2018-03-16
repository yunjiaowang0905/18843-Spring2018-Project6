import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
import os

def csv2mat(csv_path, mat_filename):
    min_time, max_time = 0, 6000
    n_lat, n_lon = 73, 18
    data = []

    with open(csv_path) as fp:
        for line in fp:
            tokens = line.splitlines()[0].split(',')
            time = int(float(tokens[0]))

            if min_time <= time and time < max_time:
                if time >= len(data):
                    data.append(np.zeros((n_lat, n_lon)))
                
                lat, lon, val = int(float(tokens[1])), int(float(tokens[2])), float(tokens[3])
                data[time][lat, lon] = val

        savemat(mat_filename, {'data_all': data})


if __name__ == "__main__":
    csv_path = '../../distribution_Res.csv'
    mat_filename = '../matdata/data_all.mat'
    csv2mat(csv_path, mat_filename)

    

