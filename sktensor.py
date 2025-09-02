import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

def two_term(A,B):
    return np.matmul(A.conj().T, B).T * np.matmul(B.conj().T, A)

def three_term(A,B,C):
    return np.matmul(A.conj().T, B).T * np.matmul(B.conj().T, C) * np.diag(np.matmul(C.conj().T, A))

def four_term(A,B,C,D):
    return np.matmul(A.conj().T, B).T * np.diag(np.matmul(B.conj().T, C)).reshape(-1,1) * np.matmul(C.conj().T, D) * np.diag(np.matmul(D.conj().T, A))

# xxx - yyy
def Cxxx(lambdaa, Uk, Uzx, Ufx):
    # C1 terms
    C1_1 = -three_term(Uk, Ufx, Uzx)
    C1_2 = -2 * two_term(Uk, Uzx)
    C1_3 = +2 * two_term(Uk, Ufx)
    C1_4 = +three_term(Uk, Uzx, Ufx)
    C1 = (1/(2*lambdaa**3)) * (C1_1 + C1_2 + C1_3 + C1_4)

    # C2 terms
    C2_1 = -three_term(Uk, Uzx, Ufx)
    C2_2 = +three_term(Uk, Ufx, Uzx)
    C2_3 = +four_term(Uk, Ufx, Uzx, Ufx)
    C2_4 = -four_term(Uk, Uzx, Ufx, Uzx)
    C2 = (1/(8*lambdaa**3)) * (C2_1 + C2_2 + C2_3 + C2_4)

    return C1 + C2


# xxy - yyx
def Cxxy(lambdaa, Uk, Uzx, Ufx, Uzy, Ufy, Uzxy, Ufxy):
    C1_1 = three_term(Uk, Uzx, Uzxy)
    C1_2 = -three_term(Uk, Ufx, Uzxy)
    C1_3 = three_term(Uk, Ufx, Uzx)
    C1_4 = -three_term(Uk, Uzx, Uzy)
    C1_5 = three_term(Uk, Ufx, Uzy)
    C1_6 = 2*two_term(Uk, Uzx)
    C1_7 = -2*two_term(Uk, Ufx)
    C1_8 = -three_term(Uk, Uzx, Ufx)
    C1_9 = -three_term(Uk, Uzx, Ufy)
    C1_10 = three_term(Uk, Ufx, Ufy)
    C1_11 = three_term(Uk, Uzx, Ufxy)
    C1_12 = -three_term(Uk, Ufx, Ufxy)

    C1 = (1/(4*lambdaa**3)) * (C1_1 + C1_2 + C1_3 + C1_4 + C1_5 + C1_6 + C1_7 + C1_8 + C1_9 + C1_10 + C1_11 + C1_12)

    C2_1 = three_term(Uk, Uzx, Uzy)
    C2_2 = -three_term(Uk, Uzx, Ufy)
    C2_3 = -four_term(Uk,Uzx,Ufx,Uzy)
    C2_4 = four_term(Uk,Uzx,Ufx,Ufy)
    C2_5 = -four_term(Uk, Ufx, Uzx, Uzy)
    C2_6 = four_term(Uk, Ufx, Uzx, Ufy)
    C2_7 = three_term(Uk, Ufx,Uzy)
    C2_8 = -three_term(Uk, Ufx, Ufy)

    C2 = (1/(8*lambdaa**3)) * (C2_1 + C2_2 + C2_3 + C2_4 + C2_5 + C2_6 + C2_7 + C2_8)

    return C1 + C2


# xyx - yxy
def Cxyx(lambdaa, Uk, Uzx, Ufx, Uzy, Ufy):
    C1_1 = three_term(Uk, Uzy, Uzx)
    C1_2 = -three_term(Uk, Ufy, Uzx)
    C1_3 = -2*two_term(Uk, Uzy)
    C1_4 = 2*two_term(Uk, Ufy)
    C1_5 = three_term(Uk, Uzy, Ufx)
    C1_6 = -three_term(Uk, Ufy, Ufx)

    C1 = (1/(2*lambdaa**3)) * (C1_1 + C1_2 + C1_3 + C1_4 + C1_5 + C1_6)

    C2_1 = -four_term(Uk, Uzy, Uzx, Ufx)
    C2_2 = four_term(Uk, Ufy, Uzx, Ufx)
    C2_3 = -four_term(Uk, Uzy, Ufx, Uzx)
    C2_4 = four_term(Uk, Ufy, Ufx, Uzx)

    C2 = (1/(8*lambdaa**3)) * (C2_1 + C2_2 + C2_3 + C2_4)


    return C1 + C2


# xyy - yxx
def Cxyy(lambdaa, Uk, Uzx, Ufx, Uzy, Ufy, Uzxy, Ufxy):

    C1_1 = three_term(Uk,Uzy,Uzxy)
    C1_2 = -three_term(Uk, Ufy, Uzxy)
    C1_3 = -three_term(Uk, Uzy, Uzx)
    C1_4 = three_term(Uk, Ufy, Uzx)
    C1_5 = three_term(Uk, Ufy, Uzy)
    C1_6 = 2*two_term(Uk, Uzy)
    C1_7 = -2*two_term(Uk, Ufy)
    C1_8 = -three_term(Uk, Uzy, Ufx)
    C1_9 = three_term(Uk, Ufy, Ufx)
    C1_10 = -three_term(Uk, Uzy, Ufy)
    C1_11 = three_term(Uk, Uzy, Ufxy)
    C1_12 = -three_term(Uk,Ufy,Ufxy)

    C1 = (1/(4*lambdaa**3)) * (C1_1 + C1_2 + C1_3 + C1_4 + C1_5 + C1_6 + C1_7 + C1_8 + C1_9 + C1_10 + C1_11 + C1_12)

    C2_1 = +four_term(Uk, Uzy, Uzx, Uzy)
    C2_2 = -four_term(Uk, Ufy, Uzx, Uzy)
    C2_3 = -four_term(Uk, Uzy, Uzx, Ufy)
    C2_4 = +four_term(Uk, Ufy, Uzx, Ufy)
    C2_5 = -four_term(Uk, Uzy, Ufx, Uzy)
    C2_6 = +four_term(Uk, Ufy, Ufx, Uzy)
    C2_7 = +four_term(Uk, Uzy, Ufx, Ufy)
    C2_8 = -four_term(Uk, Ufy, Ufx, Ufy)

    C2 = (1/(8*lambdaa**3)) * (C2_1 + C2_2 + C2_3 + C2_4 + C2_5 + C2_6 + C2_7 + C2_8)

    return C1 + C2

    
