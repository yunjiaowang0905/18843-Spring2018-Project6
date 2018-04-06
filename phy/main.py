import scipy.io as scipy
import numpy as np
import config
from sys_run import sys_run
import csv

def main():
    config.init()
    if config.run_alg > 0:
        with open(config.NEWDATA_PATH) as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                if (float(row['pm']) > 0.0):
                    config.set_x_est_gp_single(int(float(row['time'])), int(float(row['x'])), int(float(row['y'])), float(row['pm']))
        # data = scipy.loadmat(config.NEWDATA_PATH)
        # tmp = np.array(data['x_est_gp'])
        # for i in range(0, config.n_time):
        #     config.set_x_est_gp(i, np.asmatrix(tmp[i][0]))
        for i_t in range(0, config.n_time):
            sys_run(i_t)

if __name__ == "__main__":
    main()