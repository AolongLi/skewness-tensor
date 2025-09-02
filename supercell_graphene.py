import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


def BG(t,delta,N):
    h0=np.array([
        [delta,t],
        [t,-delta]
    ])

    T1=np.zeros((2,2))
    T1[1,0]=t

    T2=np.zeros((2,2))
    T2[0,1]=t

    H0_1=np.kron(np.diag(np.ones(N)),h0)
    H0_2=np.kron(np.diag(np.ones(N-1),1),T2)
    H0_3=np.kron(np.diag(np.ones(N-1),-1),T2.T.conj())
    H0=H0_1+H0_2+H0_3

    H1=np.kron(np.diag(np.ones(N)),T1) #2N

    H_1=np.kron(np.diag(np.ones(N)),H0)
    H_2=np.kron(np.diag(np.ones(N-1),1),H1)
    H_3=np.kron(np.diag(np.ones(N-1),-1),H1.T.conj())
    H=H_1+H_2+H_3 #2N**2

    Ty0=np.zeros((2*N,2*N))
    Ty0[2*N-2:2*N,:2]=T2

    Ty=np.kron(np.diag(np.ones(N)),Ty0)

    Tx=np.zeros((2*N**2,2*N**2))
    Tx[2*N**2-2*N:2*N**2,:2*N]=H1

    return H,Tx,Ty


def generate_H(kvec,t,delta,N,W,random_vector):
    H,Tx,Ty=BG(t,delta,N)

    print("H",H)
    print("Tx",np.real(Tx))
    print("Ty",np.real(Ty))
    kx=kvec[0]
    ky=kvec[1]

    x_plane_wave=np.exp(-1j*((np.sqrt(3)/2)*kx+1.5*ky))
    x_plane_wave_f=np.exp(1j*((np.sqrt(3)/2)*kx+1.5*ky))
    y_plane_wave=np.exp(-1j*((np.sqrt(3)/2)*kx-1.5*ky))
    y_plane_wave_f=np.exp(1j*((np.sqrt(3)/2)*kx-1.5*ky))

    H_on = H
    H_x_hop = Tx * x_plane_wave + Tx.T.conj() * x_plane_wave_f
    H_y_hop = Ty * y_plane_wave + Ty.T.conj() * y_plane_wave_f

    disorder=W*np.diag(random_vector) #(2*N**2)

    Htotal=H_on+disorder+H_x_hop+H_y_hop
    #print('shape:',Htotal.shape)
    return Htotal


def plt_band(N,color):
    t=-3.16
    num_points=1000
    # klist=np.linspace([-np.pi*N,0],[np.pi*N,0],num_points)
    klist=np.linspace([0,-np.pi*N],[0,np.pi*N],num_points)
    delta=0
    # N=2
    random_vector=np.zeros((2*N**2))
    W=0

    evalues = []
    for kvec in klist:
        Htotal=generate_H(kvec,t,delta,N,W,random_vector)
        evalue,evector=LA.eigh(Htotal)
        evalues.append(evalue)
    
    evalues = np.array(evalues)

    plt.plot(klist[:, 1]/N, evalues,color=color)
    plt.ylabel('meV')
    # plt.ylim([-100,100])
    plt.xlabel('k_{x}')
    plt.title('N=%d'%N)
    



plt_band(1,'blue')

plt.show()


