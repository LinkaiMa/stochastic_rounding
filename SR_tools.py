import math
import numpy as np
from matplotlib import pyplot as plt
def round_down(x,digits,base=10):
    # round down number x to d digits.
    x = np.array(x)
    flr = np.vectorize(math.floor)
    return flr(x*(base**digits))/base**digits


def SR_single (x,digits,base=10):
    R = 1/base**(digits) # gap between two numbers
    x_round = round_down(x,digits,base)
    if np.random.uniform()*R < x-x_round:
        x_round += R
    return x_round

def RN_single(x,digits,base=10):
    R = 1/base**(digits) # gap between two numbers
    x_round = round_down(x,digits,base)
    if x-x_round > R/2:
        x_round += R
    return x_round


def SR(A,digits,base=10):
    A = np.array(A)
    sr = np.vectorize(SR_single)
    return sr(A,digits,base)

def RN(A,digits,base=10):
    A = np.array(A)
    rn = np.vectorize(RN_single)
    return rn(A,digits,base)