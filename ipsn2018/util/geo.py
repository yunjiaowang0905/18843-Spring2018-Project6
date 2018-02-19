__author__ = 'Nanshu Wang'
from numpy import pi

def deg2km(degree, radius = 6371):
    """
    :param degree: distances from degrees on earth
    :return: converts distances from degrees to kilometers as measured along a great circle
    on a sphere with a radius of 6371 km, the mean radius of the Earth.
    """
    return degree * 1.0 / 360 * 2 * pi * radius