import numpy as np


def Ab(w: float, d: float, kx: float, cP: float, cS: float) -> float:
    """Characteristic equation for A modes in a free plate"""
    h = d/2
    gs = np.sqrt((w/cS)**2-kx**2 +0j)
    gp = np.sqrt((w/cP)**2-kx**2 +0j)
    return np.real(gs*np.tan(gs*h) + (gs**2 - kx**2 + 0j)**2 * np.tan(gp*h)/(4*kx**2 * gp))


def Sb(w: float, d: float, kx: float, cP: float, cS:float) -> float:
    """Characteristic equation for S modes in a free plate"""
    h = d/2
    gs = np.sqrt((w/cS)**2-kx**2 +0j)
    gp = np.sqrt((w/cP)**2-kx**2 +0j)
    return np.real(np.tan(gs*h)/gs + np.tan(gp*h)*(4*kx**2*gp)/(gs**2 - kx**2)**2)


def Ab_leaky(w: float, kx: complex, cS: float, cP: float, rho: float, rhoF: float, cF: float, h: float) -> complex:
    """Characteristic equation for A modes in a fluid-immersed plate.
    
    NOTE: This function uses the i(kx-wt) convention!
    """
    c = w / kx + 0j
    qS = np.sqrt(1 - (c / cS) ** 2)
    qP = np.sqrt(1 - (c / cP) ** 2)
    QP = kx * h * qP
    QS = kx * h * qS
    Q0 = (1 + qS ** 2) + 0j
    kS = w / cS + 0j
    kP = w / cP + 0j
    kF = w / cF + 0j
    
    part1 = Q0 ** 2 * np.tanh(QP) / np.tanh(QS) - 4 * qS * qP
    part2 = 1j * rhoF / rho * (kS / kx) ** 4 * np.sqrt((kx ** 2 - kP ** 2) / (kF ** 2 - kx ** 2)) / np.tanh(QS)
    return part1 + part2


def Sb_leaky(w: float, kx: complex, cS: float, cP: float, rho: float, rhoF: float, cF: float, h: float) -> complex:
    raise NotImplementedError()
