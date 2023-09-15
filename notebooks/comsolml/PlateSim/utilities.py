import numpy as np

pi = np.pi
nax = np.newaxis

from numpy.linalg import inv, det
from scipy.signal import hilbert


""" Matrices """

def T(gp, gs, y):
    T = np.zeros((4,4), dtype = complex)
    T[0,0] = np.exp(1j*gp*y)
    T[1,1] = np.exp(-1j*gp*y)
    T[2,2] = np.exp(1j*gs*y)
    T[3,3] = np.exp(-1j*gs*y)
    return T

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
                 [-e, -e,  f, -f]], dtype = complex)
    return M

def D(y, kx, w, Material):
    G = Material.G
    gp = np.conj(np.sqrt((w / Material.cP)**2 - kx**2 + 1e-16 + 0j))
    gs = np.conj(np.sqrt((w / Material.cS)**2 - kx**2 + 1e-16 + 0j))
    return M(kx, gs, gp, G)@T(gp,gs,y)

def L(y, kx, w, Material):
    return D(-y, kx, w, Material) @ inv(D(y, kx, w, Material))




""" Matrix applications """

def Lamb_det(h,w,kx,Material):
    mat = L(h,kx,w,Material)
    mat = mat[0:2,2:4]
    return det(mat)

def general_Lamb_det(h,w,kx, plate_mat, top_mat, bot_mat):
    mat = inv(D(0, kx, w, bot_mat)) @ L(h,kx,w,plate_mat) @ D(0, kx, w, top_mat) 
    return det(mat[1:4:2,1:4:2])

def Tp_Ts_Rp_Rs(w, kx, mid_thickness, top_mat, mid_mat, bot_mat):
    mat = inv(D(0, kx, w, bot_mat)) @ L(mid_thickness/2,kx,w,mid_mat) @ D(0, kx, w, top_mat)
    M = np.array([[1, 0, -mat[0,1], -mat[0,3]],
                  [0, 0, -mat[1,1], -mat[1,3]],
                  [0, 1, -mat[2,1], -mat[2,3]],
                  [0, 0, -mat[3,1], -mat[3,3]]]) 
    c =  np.array([mat[0,0], mat[1,0], mat[2,0],mat[3,0]])
    return inv(M)@c

def R_fluid_solid(w, kx, Fluid, Solid):
    """
    Reflection coefficient from fluid-solid interface
    - Fluid & Solid are classes
    - kx = w/cf * np.sin(thetas) 
    """
    cf = Fluid.cP
    cp = Solid.cP
    cs = Solid.cS

    w = w[:, nax]
    kx = kx[nax, :]
    gp1 = np.conj(np.sqrt((w / cf)**2 - kx**2 + 1e-16j))
    gp2 = np.conj(np.sqrt((w / cp)**2 - kx**2 + 1e-16j))
    gs2 = np.conj(np.sqrt((w / cs)**2 - kx**2 + 1e-16j))
    Zp1 = w * Fluid.rho / gp1
    Zp2 = w * Solid.rho / gp2
    Zs2 = w * Solid.rho / gs2
    cos2sq = (2*(kx/w)**2*cs**2-1)**2
    sin2sq = (2*kx*gs2*(cs/w)**2)**2
    res = (Zp2 *cos2sq + Zs2*sin2sq - Zp1)/(Zp2 *cos2sq + Zs2*sin2sq + Zp1)
    return np.squeeze(res)


def dBnorm(array, array2 = 0):
    array = np.abs(array)
    if np.isscalar(array2) == True:
        array = array / np.max(array)
    else:
        array = array / np.max(np.abs(array2))
    return 20*np.log10(array)

def norm(array, array2 = 0):
    array = np.abs(array)
    if np.isscalar(array2) == True:
        array = array / np.max(array)
    else:
        array = array / np.max(np.abs(array2))
    return (array)

def RNSE(y1, y2, yref = 0):
    if np.isscalar(yref) == True:
        if yref == 0: yref = y2
    E = np.sum( (y1-y2)**2 )
    Eref = np.sum(yref**2)
    return np.sqrt(E / Eref)
    
def cos_similarity(y, yref, give = 'cos'):
    cos_theta = np.sum(y*yref) / ( np.sqrt(np.sum(y**2)) * np.sqrt(np.sum(yref**2)) )
    if give == 'cos':
        return cos_theta
    if (give == 'angle') or (give == 'theta'):
        return np.arccos(cos_theta)*180/pi


def env(x):
    return np.abs(hilbert(x))

#def C_cos(x1, x2):
#    return 1 - np.sum(x1*x2) / ( np.sqrt(np.sum(x1**2)) * np.sqrt(np.sum(x2**2)) )

#def C_e(x1, x2):
#    E1 = np.sum(x1**2)
#    E2 = np.sum(x2**2)
#    return abs(E1-E2)/(E1+E2+1e-30)

#def cost(x1,x2):
#    return 100*((C_e(x1,x2) + 1)*(C_cos(x1,x2) + 1) - 1)/3

#def array_cost(t, xs, p, p_ref):
#    cost_sum = 0
#    for xi in xs:
#        p0 = env(p_ref(t, xi))
#        p1 = env(p(t, xi))
#        cost_sum += cost(p0,p1)
#    return cost_sum

# def array_rNSR_cost(t, xs, p, p_ref):
#     cost_sum = 0
#     for xi in xs:
#         p0 = env(p_ref(t_eval, xi))
#         p1 = env(    p(t_eval, xi))
#
#         se = np.sum((p0-p1)**2)
#         ss = np.sum(p0**2)
#
#         cost_sum += np.sqrt(se/ss)
#     return cost_sum/len(xs)
