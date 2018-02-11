import scipy.io as scipy
import numpy as np
import config

def main():
    config.init()
    if config.run_alg > 0:
        data = scipy.loadmat(config.RESULT_PATH)
        tmp = np.array(data['x_est_gp'])
        for i in range(0, n_time):
            config.set_x_est_gp(i, tmp[i][0])
        for i_t in range(0, n_time):
            sys_run(i_t)

if __name__ == "__main__":
    main()