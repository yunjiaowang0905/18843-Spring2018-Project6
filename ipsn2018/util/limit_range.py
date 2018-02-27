__author__ = 'Nanshu Wang'

def limit_range(data_in, min_val, max_val):
    """
    :param data_in: a numpy array object
    :param min_val: minimal value of the range
    :param max_val: maximal value of the range
    :return: a numpy array object where all the value is within the range limits
    """
    data_out = data_in
    data_out[data_out < min_val] = min_val
    data_out[data_out > max_val] = max_val
    return data_out