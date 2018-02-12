#!/usr/bin/env python

"""An example script to run the system.

Usage:
    run_sys.py [-h] [-1] [-2] [-3]

 Options:
    -h, --help       Show the help
    -1, --step1      Execute step1 (baseline ML algorithms)
    -2, --step2      Execute step2 (particle filter)
    -3, --step3      Execute step3 (flag learning)

"""

__author__ = 'Nanshu Wang'

import os
import sys
import yaml
import docopt

sys.path.append(os.path.join(os.path.dirname(__file__), "util"))
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from model import particle_filter, neural_network, gaussian_process

ROOT_DIR = os.path.abspath(os.path.join(__file__, os.pardir))
CONF_DIR = ROOT_DIR + "/conf"
DATA_DIR = ROOT_DIR + "/data"

if __name__ == "__main__":
    args = docopt.docopt(__doc__)
    execute_steps = [False] \
        + [args["--step{}".format(step_index)] for step_index in range(1, 4)]

    if not any(execute_steps):
        print 'Please specify which step to execute. Try "run_sys.py -h" for help. '

    with open(CONF_DIR + "/default.yml", "r") as f:
        conf = yaml.load(f)

    if execute_steps[1]:
        print '### 1. run baseline ML algorithms ###'
        nn = neural_network(conf)
        gp = gaussian_process(conf)

    if execute_steps[2]:
        print '### 2. run particle filter algorithm ###'
        pf = particle_filter(conf)
        pf.predict()

    if execute_steps[3]:
        print '### 3. run flag learn ###'
        pass

