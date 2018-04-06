from __future__ import absolute_import, division, print_function
__author__ = 'Nanshu Wang'

def predictFlag(history, new):
    l = len(history)
    weight = [(i+1)/l/2 for i in range(l)]
    weight = weight + [1]
    weight = [v/sum(weight) for v in weight]
    weightedAvg = new * weight[-1]
    for i in range(l):
        weightedAvg = weightedAvg + weight[i] * history[i]
    return weightedAvg