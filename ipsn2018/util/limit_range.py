__author__ = 'Nanshu Wang'

def limit_range(data_in, min_val, max_val):
    data_out = data_in
    data_out[data_out < min_val] = min_val
    data_out[data_out > max_val] = max_val
    return data_out