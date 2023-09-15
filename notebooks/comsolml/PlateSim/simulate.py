from typing import Type
import numpy as np
from numpy import fft
from scipy import interpolate
from scipy import optimize as opt
from .Lamb_waves import Lamb, Lamb_A, Lamb_S
from .utilities import Tp_Ts_Rp_Rs
from .plate import Plate, ImmersedPlate
from .materials import FluidMaterial, ElasticMaterial
from .characteristic_equations import Ab_leaky, Sb_leaky

pi = np.pi
nax = np.newaxis

## COMMON
def make_pulse_fast(pAmplitude, fluid, x, t, fc = 250000, deg = 32, width = 5, length = 2, x0 = -1, y0 = -1, n_len = 10, n_time =2):
    
    # Pulse parameters
    wc       = 2*pi*fc
    kc       = wc/fluid.cP
    lambda_c = 2*pi/kc
    
    pulse_width = width*lambda_c / 2.355
    pulse_length = length*lambda_c / 2.355
    
    th = deg*pi/180 # direction
    kc_x = +kc * np.sin(th)
    kc_y = -kc * np.cos(th)

    # Choose a position if none are given based on wave packet size
    if (x0 == -1) or (y0 == -1):
        x0 = y0 = 2.5*np.sqrt(pulse_width**2 + pulse_length**2) 

    # Simulation domain & discretization    
    dx = x[1] - x[0] 
    y = np.copy(x)
    
    # Crop domain to **significantly** reduce computation time
    crop_len = np.sqrt(width**2 + length**2) * lambda_c/2.355*n_len
    
    ix = np.where((x > (x0 - crop_len)) & (x < (x0 + crop_len)))
    xcrop = x[ix]
    ycrop = y[ (y >= 0) & (y < (y0 + crop_len)) ]

    crop_t = n_time*(abs(y0/np.cos(th)) + pulse_length ) / fluid.cP
    it = np.where(t <= crop_t)
    tcrop = t[it]
    
    # Calculate initial vertical velocity distribution, spatial and complex Gaussian pulse
    xx, yy = np.meshgrid(xcrop, ycrop)

    ang = pi/2-th
    a = np.cos(ang)**2 / (2*pulse_length**2) + np.sin(ang)**2 /(2*pulse_width**2)
    b = -np.sin(2*ang) / (4*pulse_length**2 ) + np.sin(2*ang) / (4*pulse_width**2)
    c = np.sin(ang)**2 /(2*pulse_length**2 ) + np.cos(ang)**2 /(2*pulse_width**2)
    gauss_window = pAmplitude * np.exp(-(a*(xx-x0)**2 + 2*b*(xx-x0)*(yy-y0) + c*(yy-y0)**2))

    p0 = gauss_window * np.exp(-1j*(kc_x*(xx-x0) + kc_y*(yy-y0)))

    # Calculate propagation at y=0 from frequency domain representation 
    P  =  fft.fft2(p0)
    k_x = 2*pi*fft.fftfreq(len(xcrop), dx)
    k_y = 2*pi*fft.fftfreq(len(ycrop), dx)
    kxx, kyy = np.meshgrid(k_x, k_y)
    k_mask = np.sqrt(kxx**2 + kyy**2)
    w_mask = k_mask * fluid.cP
    p = fft.ifftn(P[nax,:,:] * np.exp(1j*w_mask[nax,:,:]*tcrop[:,nax,nax]), axes =(1,2))
    p_line = (p[:,0, : ])
    
    # zero-pad to wanted lengths in x and t
    pad_t_top = len(t)-it[0][-1] - 1
    pad_x_left = ix[0][0]
    pad_x_right = len(x) - ix[0][-1] - 1
    p_line_pad = np.pad(p_line, ((0,pad_t_top),(pad_x_left, pad_x_right)), 'constant')

    return p_line_pad, p0

def make_pulse(pAmplitude,fluid, x, t, fc = 250000, deg = 32, width = 5, length = 2, x0 = -1, y0 = -1, phase = 0):
    
    # Pulse parameters
    wc = 2*pi*fc
    kc = wc/fluid.cP
    lambda_c = 2*pi/kc
    
    pulse_width = width*lambda_c / 2.355
    pulse_length = length*lambda_c / 2.355

    if (x0 == -1) or (y0 == -1):
        x0 = y0 = 2.5*np.sqrt(pulse_width**2 + pulse_length**2) 

    th = deg*pi/180 # direction
    kc_x = +kc * np.sin(th)
    kc_y = -kc * np.cos(th)

    # Simulation domain & discretization    
    dx = x[1] - x[0] # to get exact numerics
    y = np.copy(x)

    # Calculate initial vertical velocity distribution, spatial and complex Gaussian pulse
    xx, yy = np.meshgrid(x, y)

    ang = pi/2-th
    a = np.cos(ang)**2 / (2*pulse_length**2) + np.sin(ang)**2 /(2*pulse_width**2)
    b = -np.sin(2*ang) / (4*pulse_length**2 ) + np.sin(2*ang) / (4*pulse_width**2)
    c = np.sin(ang)**2 /(2*pulse_length**2 ) + np.cos(ang)**2 /(2*pulse_width**2)
    gauss_window = pAmplitude * np.exp(-(a*(xx-x0)**2 + 2*b*(xx-x0)*(yy-y0) + c*(yy-y0)**2))

    p0 = gauss_window * np.exp(-1j*(kc_x*(xx-x0) + kc_y*(yy-y0)))
    if phase != 0:
        p0 = p0*phase

    # Calculate the frequency domain representation 
    P  =  fft.fft2(p0)

    k_x = 2*pi*fft.fftfreq(len(x), dx)
    k_y = 2*pi*fft.fftfreq(len(x), dx)
    kxx, kyy = np.meshgrid(k_x, k_y)
    k_mask = np.sqrt(kxx**2 + kyy**2)
    w_mask = k_mask * fluid.cP
    p = fft.ifftn(P[nax,:,:] * np.exp(1j*w_mask[nax,:,:]*t[:,nax,nax]), axes =(1,2))
    p_line = (p[:,0, : ])
    return p_line, p0

    # p_line, p0 = make_pulse(water2, x, t, fc = 250000, deg = 32, width = 5, length = 2)
    # plt.pcolormesh(np.real(p_line))
    # plt.axis('equal')

def sample_dimensions_const_max(fc, c_min, x_max, t_max, samples_per_wavelength = 5, samples_per_f = 5, pow2 = True):
    wavelength = c_min /fc
    n_x = int(x_max / wavelength * samples_per_wavelength)
    if pow2 == True:
        n_x = int(2**(np.ceil(np.log2(n_x))))
    x = np.linspace(0, x_max, n_x, endpoint = False)

    n_t = int(t_max * fc * samples_per_f)
    if pow2 == True:
        n_t = int(2**(np.ceil(np.log2(n_t))))
    t = np.linspace(0, t_max, n_t, endpoint = False)
    return x, t

def sample_dimensions_const_delta(fc, c_min, x_max, t_max, samples_per_wavelength = 5, samples_per_f = 5, pow2 = True):
    wavelength = c_min /fc
    dx = wavelength / samples_per_wavelength
    n_x = int(x_max/dx)
    if pow2 == True:
        n_x = int(2**(np.ceil(np.log2(n_x))))
        x_max = dx*n_x
    x = np.linspace(0, x_max, n_x, endpoint = False)

    dt = 1/(fc*samples_per_f)
    n_t = int(t_max/dt)
    if pow2 == True:
        n_t = int(2**(np.ceil(np.log2(n_t))))
        t_max = dt*n_t
    t = np.linspace(0, t_max, n_t, endpoint = False)
    return x, t

# RESPONSE 

def create_interpolation(P,t,x,n):
    p = np.real(fft.ifft2(P))
    p = fft.fftshift( p, axes = 1 )
    p = np.roll(p, -n, axis = 1)
    interp = interpolate.RectBivariateSpline(t,x, p)
    return interp.ev


def compute_responses(params, Ra, Ta, Rb, D, V, n_responses = 4, total = True):
    x = params['x']
    t = params['t']
    P_in = params['P_in']
    n = params['n']
    
    P_0 = Ra * P_in
    
    if total == False:
        p_tx = []
        interp = create_interpolation(P_0,t,x,n)
        p_tx.append(interp)
    
    Y = D * V( Rb * V( D * Ta * P_in ) )
    for _ in range(n_responses):
        P_i =  Ta * Y
        
        if total == False:
            interp = create_interpolation(P_i,t,x,n)
            p_tx.append(interp)
        P_0 += P_i

        Y = D * V( Rb * V( D * Ra * Y ) )
    
    if total == True: 
        p_tx = create_interpolation(P_0,t,x,n)
    return p_tx

def R_T_plate(w, kx, plate_material, fluid, d = 0.01):
    R = np.zeros((w.shape[0], kx.shape[0]), dtype=np.complex)
    T = np.zeros((w.shape[0], kx.shape[0]), dtype=np.complex)

    wend = len(w)//2+1 # kan være mikrofeil her med å speile under
    kend = len(kx)//2+1

    for i, wi in enumerate(w[0:wend]):
        for j, kxj in enumerate(kx[0:kend]):
            if abs(wi)/fluid.cP <= abs(kxj): continue
            if abs(wi) < 1000: continue
            T[i,j], _, R[i,j], _ = Tp_Ts_Rp_Rs(wi, kxj, d, fluid, plate_material, fluid)

    
    if len(w)%2==0:
        T[wend:, :] = T[wend-2:0:-1, :]
        R[wend:, :] = R[wend-2:0:-1, :]
    else:
        T[wend:, :] = T[wend-1:0:-1, :]
        R[wend:, :] = R[wend-1:0:-1, :]
    
    if len(kx)%2==0:
        T[:, kend:] = T[:, kend-2:0:-1]
        R[:, kend:] = R[:, kend-2:0:-1]
    else:
        T[:, kend:] = T[:, kend-1:0:-1]
        R[:, kend:] = R[:, kend-1:0:-1]

    T[np.isnan(T)] = 0
    R[np.isnan(R)] = 0
    T[abs(T)>1] = 1
    R[abs(R)>1] = 1

    return R, T


def R_T_plate_general(w, kx, top, plate_material, bot, d = 0.01):
    R = np.zeros((w.shape[0], kx.shape[0]), dtype=np.complex)
    T = np.zeros((w.shape[0], kx.shape[0]), dtype=np.complex)

    wend = len(w)//2+1 # kan være mikrofeil her med å speile under
    kend = len(kx)//2+1

    for i, wi in enumerate(w[0:wend]):
        for j, kxj in enumerate(kx[0:kend]):
            if abs(wi)/top.cP <= abs(kxj): continue
            if abs(wi) < 1000: continue
            T[i,j], _, R[i,j], _ = Tp_Ts_Rp_Rs(wi, kxj, d, top, plate_material, bot)

    
    if len(w)%2==0:
        T[wend:, :] = T[wend-2:0:-1, :]
        R[wend:, :] = R[wend-2:0:-1, :]
    else:
        T[wend:, :] = T[wend-1:0:-1, :]
        R[wend:, :] = R[wend-1:0:-1, :]
    
    if len(kx)%2==0:
        T[:, kend:] = T[:, kend-2:0:-1]
        R[:, kend:] = R[:, kend-2:0:-1]
    else:
        T[:, kend:] = T[:, kend-1:0:-1]
        R[:, kend:] = R[:, kend-1:0:-1]

    T[np.isnan(T)] = 0
    R[np.isnan(R)] = 0
    T[abs(T)>1] = 1
    R[abs(R)>1] = 1

    return R, T

def R(w, kx, fluid, solid):
    cf = fluid.cP
    p_fluid = fluid.rho
    p_solid = solid.rho
    cp = solid.cP
    cs = solid.cS


    w = w[:, nax]
    kx = kx[nax, :]
    gp1 = np.conj(np.sqrt((w / cf)**2 - kx**2 + 1e-16j))
    gp2 = np.conj(np.sqrt((w / cp)**2 - kx**2 + 1e-16j))
    gs2 = np.conj(np.sqrt((w / cs)**2 - kx**2 + 1e-16j))
    Zp1 = w * p_fluid / gp1
    Zp2 = w * p_solid / gp2
    Zs2 = w * p_solid / gs2
    cos2sq = (2*(kx/w)**2*cs**2-1)**2
    sin2sq = (2*kx*gs2*(cs/w)**2)**2
    res = (Zp2 *cos2sq + Zs2*sin2sq - Zp1)/(Zp2 *cos2sq + Zs2*sin2sq + Zp1)
    return np.squeeze(res)

def Tilt(X, deg_tilt, kx, w, cf):
    tilt = deg_tilt * pi/180
    
    ky = np.conj(np.sqrt((w[:, nax]/cf)**2 - (kx[nax, :])**2 + 0j))
    mask = abs(np.real(ky)) >= 1e-2 # outside of which waves are evanescent, division by near 0.

    F = np.zeros_like(X) # X.shape has dimensions w-k
    for i, wi in enumerate(w):
        Gd = interpolate.interp1d(kx, X[i,:], kind = 'linear', fill_value=0,bounds_error=False)
        kx_int = kx*np.cos(tilt) - np.real(ky[i,:])*np.sin(tilt) 
        F[i,:] = Gd(kx_int) * np.abs( np.cos(tilt) + kx/(ky[i,:] + 1e-15)*np.sin(tilt) )
    F = F*mask
    
    return F

# MODAL

def make_1D_gaussian_wavepacket(mode, vAmplitude, x, fc = 250000, x0 = 0, width = 2, phase = 0):
    
    # Pulse parameters
    wc = 2*pi*fc
    kc = mode.kx_from_w(wc)
    lambda_c = 2*pi/kc
    
    pulse_width = width*lambda_c / 2.355 # fwhm = width * wavelength

    # Simulation domain & discretization    
    dx = x[1] - x[0] # to get exact numerics
    kx = 2*pi*fft.fftfreq(len(x), dx)
    
    gauss_window = vAmplitude * np.exp( -(x-x0)**2 / (2*pulse_width**2) )
    v0 = gauss_window * np.exp( -1j*kc*(x-x0) )
    if phase != 0:
        v0 = v0*phase
    
    return v0, kx

def alpha_Merkulov_Antisymmetric(wave, solid, d, fluid):
    w = wave.w + 1e-10 # angular frequency
    kx = wave.kx + 1e-10# wavenumber of plate wave
    h = d/2 # half-thickness
    
    kS = w/solid.cS # shear wavenumber
    kf = w/fluid.cP # wavenumber in fluid
    qS = np.sqrt(kx**2 - (w/solid.cS)**2 + 0j)/kx # vertical S-wave number, scaled with i
    qP = np.sqrt(kx**2 - (w/solid.cP)**2 + 0j)/kx # vertical P-wave number, scaled with i
    QS  = qS*kx*h
    QP = qP*kx*h
    
    part1 = h*( kx/qP * (np.tanh(QP) - 1/np.tanh(QP)) - kx/qS * (np.tanh(QS) - 1/np.tanh(QS)) )
    part2 = 1/qP**2 + 1/qS**2 - 2*(3 - qS**2)/(1 + qS**2)
    part3 = (fluid.rho * kS**4)/(4*solid.rho*kx**2 * qS * np.sqrt((kf)**2 - kx**2 + 0j) * np.tanh(QS))
    
    return np.real(part3/(part1 + part2))


def two_sided_attenuation(wave, material, d, fluid):
    vy = wave.vy(y=np.array([d/2]))
    Ptot = wave.Px()
    
    vys_rms = (1/np.sqrt(2))*np.abs(vy)
    Zw = fluid.cP*fluid.rho
    Plost = 2 * np.real((Zw * vys_rms**2) / np.sqrt(1-(fluid.cP/wave.c)**2 + 0j + 1e-10))
    
    alpha = Plost / (2*Ptot)
    return alpha


def calculate_fluid_attenuation(w: np.ndarray, kx: np.ndarray,
                                plate: Plate, fluid: FluidMaterial,
                                w_or_k_or_alpha: str = 'w', sym: str = 'A'):
    lamb_cls: Type[Lamb]   # Specify that lamb_cls refers to Lamb (or a subclass thereof)
    if sym.upper() == 'A':
        lamb_cls = Lamb_A
    elif sym.upper() == 'S':
        lamb_cls = Lamb_S
    else:
        raise ValueError("sym argument must be 'A' or 'S'")
    
    alphas = np.zeros_like(w)
    res = np.zeros_like(w, dtype='complex')
    
    for i, wi in enumerate(w):
        if abs(wi) < 2*pi*500: continue
        wave = lamb_cls(plate, kx = abs(kx[i]), w = abs(wi))
        alphas[i] = two_sided_attenuation(wave, plate.material, plate.d, fluid)
    
        if w_or_k_or_alpha == 'w':
            res[i] = wi + 1j*alphas[i]*wave.c
        
        elif w_or_k_or_alpha == 'k':
            res[i] = kx[i] - 1j*alphas[i]
    
    if w_or_k_or_alpha == 'alpha':
        return alphas
    else:
        return res


def calculate_solid_attenuation(w: np.ndarray, kx: np.ndarray,
                                plate: Plate, solid: ElasticMaterial,
                                w_or_k_or_alpha: str = 'w', sym: str = 'A'):
    lamb_cls: Type[Lamb]   # Specify that lamb_cls refers to Lamb (or a subclass thereof)
    if sym.upper() == 'A':
        lamb_cls = Lamb_A
    elif sym.upper() == 'S':
        lamb_cls = Lamb_S
    else:
        raise ValueError("sym argument must be 'A' or 'S'")
    
    alphas = np.zeros_like(w)
    res = np.zeros_like(w, dtype='complex')
    
    for i, wi in enumerate(w):
        if abs(wi) < 2*pi*500: continue
        wave = lamb_cls(plate, kx = abs(kx[i]), w = abs(wi))
        alphas[i] = two_sided_attenuation_solid(wave, plate.material, plate.d, solid)
    
        if w_or_k_or_alpha == 'w':
            res[i] = wi + 1j*alphas[i]*wave.c
        
        elif w_or_k_or_alpha == 'k':
            res[i] = kx[i] - 1j*alphas[i]
            
    if w_or_k_or_alpha == 'alpha':
        return alphas
    else:
        return res


def two_sided_attenuation_solid(wave, material, d, solid):
    vx = wave.vx(y=np.array([d/2]))
    vy = wave.vy(y=np.array([d/2]))
    Px = wave.Px()
    
    # wave numper in solid
    kp = wave.w/solid.cP
    ks = wave.w/solid.cS
    
    # beta = horizontal wavenumber (same in both plate and solid). Can be imaginary!
    beta = wave.kx + 0j
    
    # Vertial wavenumbers in solid
    gp = np.conj(np.sqrt(kp**2 - beta**2 + 0.0j)) # Ikke konjugert her, antar jeg... eksponensielt _økende_ utover...
    gs = np.conj(np.sqrt(ks**2 - beta**2 + 0.0j))

    # ampitude of P- and S-wave
    vp = kp * (gs*vy + beta*vx)  / (beta**2 + (gs*gp)) 
    vs = ks * (-beta*vy + gp*vx) / (beta**2 + (gs*gp)) 
 
    vp_rms = (1/np.sqrt(2))*np.abs(vp)
    vs_rms = (1/np.sqrt(2))*np.abs(vs)
    Zp = solid.cP*solid.rho
    Zs = solid.cS*solid.rho
    
    I_p = 2 * np.real((Zp * vp_rms**2) * np.sqrt(1-(beta/kp)**2+0.0j))
    I_s = 2 * np.real((Zs * vs_rms**2) * np.sqrt(1-(beta/ks)**2+0.0j))

    alpha = (I_p + I_s) / (2*Px)
    return alpha
