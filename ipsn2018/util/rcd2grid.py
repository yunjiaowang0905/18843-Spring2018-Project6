__author__ = 'Jin Gong'
import numpy as np

def rcd2grid(data_rcd, n_lat, n_lon, flag_empty):
    """
    transfer recording data into grid and select the gas data needed
    return: data_grid: gas data in each grid
             smp_cnt: number of sample in each grid
    parameters: data_rcd: data from records
                alg_grid: method to combine data in each grid
                n_lat, n_lon: latitude and longitude grids number
                i_gas: gas index, co 3, co2 4, o3 5, pm25 6
    """
    i_gas = 5
    smp_cnt = np.zeros((n_lat, n_lon))
    data_grid = [[None for i in range(n_lon)] for j in range(n_lat)]
    if not flag_empty:
        for i_x in range(n_lon):
            for i_y in range(n_lat):
                data_rcd_x_y = [rcd_x_y for rcd_x_y in data_rcd if rcd_x_y[2] == i_y and rcd_x_y[3] == i_x]
                if data_rcd_x_y:
                    print("rcd running: " + str(i_x * i_y))
                    smp_cnt[i_y][i_x] = len(data_rcd_x_y)
                    data_grid[i_y][i_x] = [row[i_gas - 1] for row in data_rcd_x_y] #data_rcd_x_y[:, i_gas - 1]
    return data_grid, smp_cnt
