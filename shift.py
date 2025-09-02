import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from skewness_tensor import *
from metric_tensor import *
from app import *
#用于计算一个k点的shift current 贡献

#单位 \frac{2\pi e^3 a}{\hbar^2 \delta}

def shiftxxx(omega,E,Ef,lambdaa, Uk, Uzx, Ufx,Sint,gamma,energy_scale):
    
    basissize=Uk.shape[1]

    # 计算skewness tensor
    print("calculate skewness tensor Cxxx...")
    cxxx=Cxxx(lambdaa, Uk, Uzx, Ufx)
    print("get skewness tensor Cxxx!")

    T = 0
    fermi = fermi(E, T, Ef)
    delta_fermi = fermi.reshape(basissize,1) - fermi.reshape(1,basissize)
    delta_E = E.reshape(basissize,1) - E.reshape(1,basissize)
    delt_func = zero_delta(omega, delta_E, gamma)

    term = cxxx * delta_fermi * delt_func
    result_at_k = (Sint)*np.sum(term)*(energy_scale)
    return result_at_k


def shiftxxy(omega,E,Ef,lambdaa, Uk, Uzx, Ufx,Uzy, Ufy,Uzxy, Ufxy,Sint,gamma,energy_scale):

    basissize=Uk.shape[1]

    print("calculate skewness tensor Cxxy...")
    cxxy=Cxxy(lambdaa, Uk, Uzx, Ufx, Uzy, Ufy, Uzxy, Ufxy)
    print("get skewness tensor Cxxy!")

    T = 0
    fermi = fermi(E, T, Ef)
    delta_fermi = fermi.reshape(basissize,1) - fermi.reshape(1,basissize)
    delta_E = E.reshape(basissize,1) - E.reshape(1,basissize)
    delt_func = zero_delta(omega, delta_E, gamma)

    term = cxxy * delta_fermi * delt_func
    result_at_k = (Sint)*np.sum(term)*(energy_scale)
    return result_at_k


def shiftxyx(omega,E,Ef,lambdaa, Uk, Uzx, Ufx,Uzy, Ufy,Sint,gamma,energy_scale):

    basissize=Uk.shape[1]

    print("calculate skewness tensor Cxyx...")
    cxyx=Cxyx(lambdaa, Uk, Uzx, Ufx, Uzy, Ufy)
    print("get skewness tensor Cxyx!")

    T = 0
    fermi = fermi(E, T, Ef)
    delta_fermi = fermi.reshape(basissize,1) - fermi.reshape(1,basissize)
    delta_E = E.reshape(basissize,1) - E.reshape(1,basissize)
    delt_func = zero_delta(omega, delta_E, gamma)

    term = cxyx * delta_fermi * delt_func
    result_at_k = (Sint)*np.sum(term)*(energy_scale)
    return result_at_k

def shiftxyy(omega,E,Ef,lambdaa, Uk, Uzx, Ufx,Uzy, Ufy,Uzxy, Ufxy,Sint,gamma,energy_scale):
    
    basissize=Uk.shape[1]

    print("calculate skewness tensor Cxyy...")
    cxyy=Cxyy(lambdaa, Uk, Uzx, Ufx, Uzy, Ufy, Uzxy, Ufxy)
    print("get skewness tensor Cxyy!")

    T = 0
    fermi = fermi(E, T, Ef)
    delta_fermi = fermi.reshape(basissize,1) - fermi.reshape(1,basissize)
    delta_E = E.reshape(basissize,1) - E.reshape(1,basissize)
    delt_func = zero_delta(omega, delta_E, gamma)

    term = cxyy * delta_fermi * delt_func
    result_at_k = (Sint)*np.sum(term)*(energy_scale)
    return result_at_k
