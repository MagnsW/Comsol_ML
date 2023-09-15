import numpy as np
import numba
pi = np.pi
nax = np.newaxis
from numpy.linalg import inv, det

""" Matrices """


@numba.njit
def T(gp, gs, y):
    T = np.zeros((4,4), dtype = np.complex128)
    T[0,0] = np.exp(1j*gp*y)
    T[1,1] = np.exp(-1j*gp*y)
    T[2,2] = np.exp(1j*gs*y)
    T[3,3] = np.exp(-1j*gs*y)
    return T

@numba.njit
def M(kx, gs, gp, G):
    a = G*(kx**2 - gs**2)
    b = 2*G*kx*gs
    c = 2*G*kx*gp
    d = 1j*gp
    e = 1j*kx
    f = 1j*gs
    M= np.array([[ a,  a, -b,  b],
                 [ c, -c,  a,  a],
                 [ d, -d,  e,  e],
                 [-e, -e,  f, -f]], dtype = np.complex128)
    return M

@numba.njit
def D(y, kx, w, mat_G, mat_cP, mat_cS):
    gp = np.conj(np.sqrt((w / mat_cP)**2 - (kx)**2  + 1e-16 + 0j))
    gs = np.conj(np.sqrt((w / mat_cS)**2 - (kx)**2  + 1e-16 + 0j)) # np.conj
    return M(kx, gs, gp, mat_G)@T(gp,gs,y)

@numba.njit
def L(y, kx, w, mat_G, mat_cP, mat_cS):
    return D(-y, kx, w, mat_G, mat_cP, mat_cS) @ inv(D(y, kx, w, mat_G, mat_cP, mat_cS))

@numba.njit
def Tp_Ts_Rp_Rs(w, kx, mid_thickness, top_G, top_cP, top_cS, mid_G, mid_cP, mid_cS,bot_G, bot_cP, bot_cS):
    mat = inv(D(0, kx, w, bot_G, bot_cP, bot_cS)) @ L(mid_thickness/2,kx,w, mid_G, mid_cP, mid_cS) @ D(0, kx, w, top_G, top_cP, top_cS)
    M = np.array([[1, 0, -mat[0,1], -mat[0,3]],
                  [0, 0, -mat[1,1], -mat[1,3]],
                  [0, 1, -mat[2,1], -mat[2,3]],
                  [0, 0, -mat[3,1], -mat[3,3]]]) 
    c =  np.array([mat[0,0], mat[1,0], mat[2,0],mat[3,0]])
    return inv(M)@c


@numba.njit(parallel = True)
def njit_R_T_general(w, kx, top_G, top_cP, top_cS, plate_G, plate_cP, plate_cS, bot_G, bot_cP, bot_cS, d):
    Rp = np.zeros((w.shape[0], kx.shape[0]), dtype = np.complex128)
    Tp = np.zeros((w.shape[0], kx.shape[0]), dtype = np.complex128)

    wend = len(w)//2+1
    kend = len(kx)//2+1

    for i in numba.prange(wend):
        for j in numba.prange(kend):
            if abs(w[i])/top_cP <= abs(kx[j]): continue
            if abs(w[i]) < 1000: continue
            Tp[i,j], _, Rp[i,j], _ = Tp_Ts_Rp_Rs(w[i], kx[j], d, top_G, top_cP, top_cS, plate_G, plate_cP, plate_cS, bot_G, bot_cP, bot_cS)

    if len(w)%2==0:
        Tp[wend:, :] = Tp[wend-2:0:-1, :]
        Rp[wend:, :] = Rp[wend-2:0:-1, :]
    else:
        Tp[wend:, :] = Tp[wend-1:0:-1, :]
        Rp[wend:, :] = Rp[wend-1:0:-1, :]
    
    if len(kx)%2==0:
        Tp[:, kend:] = Tp[:, kend-2:0:-1]
        Rp[:, kend:] = Rp[:, kend-2:0:-1]
    else:
        Tp[:, kend:] = Tp[:, kend-1:0:-1]
        Rp[:, kend:] = Rp[:, kend-1:0:-1]
    
    return Rp, Tp

def jit_R_T_plate_general(w, kx, top, plate, bot, d = 0.01):
    Rp,Tp = njit_R_T_general(w, kx, top.G, top.cP, top.cS, plate.G, plate.cP, plate.cS, bot.G, bot.cP, bot.cS, d)
    Tp[np.isnan(Tp)] = 0
    Rp[np.isnan(Rp)] = 0
    Tp[abs(Tp)>1] = 1
    Rp[abs(Rp)>1] = 1
    return Rp, Tp




# Char eq.

@numba.njit
def D2(y, kx, w, mat_G, mat_cP, mat_cS):
    gp = (np.sqrt((w / mat_cP)**2 - (kx)**2  + 0j))
    gs = (np.sqrt((w / mat_cS)**2 - (kx)**2  + 0j)) 
    return M(kx, gs, gp, mat_G)@T(gp,gs,y)

@numba.njit
def L2(y, kx, w, mat_G, mat_cP, mat_cS):
    return D2(-y, kx, w, mat_G, mat_cP, mat_cS) @ inv(D2(y, kx, w, mat_G, mat_cP, mat_cS))



@numba.njit
def jit_general_Lamb_det(h, w, kx, top_G, top_cP, top_cS, plate_G, plate_cP, plate_cS, bot_G, bot_cP, bot_cS):
    mat = inv(D2(0, kx, w, bot_G, bot_cP, bot_cS)) @ L2(h,kx,w, plate_G, plate_cP, plate_cS) @ D2(0, kx, w, top_G, top_cP, top_cS) 
    return det(mat[1:4:2,1:4:2])

def jit_Lamb_det(w, kx, top_mat, plate_mat, bot_mat, d=0.01):
    return jit_general_Lamb_det(d/2, w, kx, top_mat.G, top_mat.cP, top_mat.cS, plate_mat.G, plate_mat.cP, plate_mat.cS, bot_mat.G, bot_mat.cP, bot_mat.cS)