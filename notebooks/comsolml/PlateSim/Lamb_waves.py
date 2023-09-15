from typing import Optional, Union, Tuple, Sequence
from abc import ABC, abstractmethod
from copy import copy
import numpy as np
import logging
from numpy.lib import scimath as sm
from scipy import integrate
from scipy import optimize as opt
from .plate import Plate, ImmersedPlate
from .characteristic_equations import Ab, Sb, Ab_leaky
from .materials import FluidMaterial

pi = np.pi
nax = np.newaxis

coordinates = Union[np.ndarray, float]


class Lamb(ABC):
    """Abstract base class for antisymmetric and symmetric Lamb waves. See subclasses for documentation."""
    def __init__(self, plate: Plate,
                 kx: complex, f: Optional[float] = None, w: Optional[float] = None,
                 kx_wt_convention: bool = True) -> None:
        self.plate = plate
        self.rho = plate.material.rho
        self.cP = plate.material.cP
        self.cS = plate.material.cS
        self.G = plate.material.G
        self.d = plate.d
        self.kx = kx
        if f is not None and w is not None:
            raise ValueError('Cannot specify both f and w at the same time')
        elif f is None and w is None:
            raise ValueError('Must specify f or w')
        elif f is not None:
            self.f = f
            self.w = 2 * pi * f
        elif w is not None:
            self.w = w
            self.f = w / (2 * pi)
        self.kx_wt_convention = kx_wt_convention

        self.h = self.d / 2
        self.c = self.w / (np.real(self.kx) + 1e-10)   # type: ignore
        self.K = 1
        
        self.gs: complex
        self.gp: complex
        self.R: complex
        self.update_g()
    
    def update_g(self) -> None:
        """Calculate the vertical wavenumbers for P and S waves"""
        self.gp = np.sqrt((self.w / self.cP) ** 2 - self.kx ** 2 + 0j)
        self.gs = np.sqrt((self.w / self.cS) ** 2 - self.kx ** 2 + 0j)

    @abstractmethod
    def update_R(self, R: Optional[complex] = None) -> None:
        """Calculate the wave combination amplitude ratio; this is done differently for A and S waves
        
        Parameters:
            R: A specified value of R to update with. If not provided, calculate R for a free plate.
        """
        ...
    
    def update_kx(self, kx: complex, R: Optional[complex] = None) -> 'Lamb':
        """Set a new value for the wavenumber and update dependent variables accordingly"""
        self.kx = kx
        self.update_g()
        self.update_R(R)
        return self
    
    @staticmethod
    def _initialize_coordinate_arrays(x: coordinates,
                                      y: coordinates,
                                      t: coordinates) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extend x, y, and t to three axes to facilitate broadcasting"""
        x_new: np.ndarray = np.atleast_1d(x)[:, np.newaxis, np.newaxis]
        y_new: np.ndarray = np.atleast_1d(y)[np.newaxis, :, np.newaxis]
        t_new: np.ndarray = np.atleast_1d(t)[np.newaxis, np.newaxis, :]
        return x_new, y_new, t_new
    
    @abstractmethod
    def ux(self, x: coordinates = 0, y: coordinates = 0, t: coordinates = 0):
        """Get the x-displacement"""
        ...
    
    @abstractmethod
    def uy(self, x: coordinates = 0, y: coordinates = 0, t: coordinates = 0):
        """Get the y-displacement"""
        ...
    
    def vx(self, x: coordinates = 0, y: coordinates = 0, t: coordinates = 0):
        """Get the x-velocity"""
        if self.kx_wt_convention:
            return - 1j*self.w * self.ux(x, y, t)
        else:
            return 1j*self.w * self.ux(x, y, t)
    
    def vy(self, x: coordinates = 0, y: coordinates = 0, t: coordinates = 0):
        """Get the y-velocity"""
        if self.kx_wt_convention:
            return - 1j*self.w * self.uy(x, y, t)
        else:
            return 1j*self.w * self.uy(x, y, t)
    
    @abstractmethod
    def N(self):
        """Calculate a normalisation factor. TODO: Consider removing this"""
        ...
    
    @abstractmethod
    def sigma_xx(self, x: coordinates = 0, y: coordinates = 0, t: coordinates = 0):
        """Get the xx-stress"""
        ...
    
    @abstractmethod
    def sigma_xy(self, x: coordinates = 0, y: coordinates = 0, t: coordinates = 0):
        """Get the xy-stress"""
        ...
    
    @abstractmethod
    def sigma_yy(self, x: coordinates = 0, y: coordinates = 0, t: coordinates = 0):
        """Get the yy-stress"""
        ...
    
    def p(self, x: coordinates = 0, y: coordinates = 0, t: coordinates = 0):
        """Get the fluid pressure above or below the plate"""
        x, y, t = self._initialize_coordinate_arrays(x, y, t)
        
        assert isinstance(self.plate, ImmersedPlate)
        if all(y > 0):
            y0 = self.h
            fluid = self.plate.top_material
        elif all(y < 0):
            y0 = - self.h
            fluid = self.plate.bottom_material
        else:
            raise RuntimeError('All y values must be either above or below the plate!')
        vy_surface = self.vy(x, np.array([y0]), t)
        ky = np.sign(y0) * sm.sqrt((self.w/fluid.cP)**2 - self.kx**2)
        p0 = fluid.rho*self.w*vy_surface / ky
        return np.squeeze(p0 * np.exp(1j*(self.w*t - self.kx*x - ky*(y-y0))))
    
    @abstractmethod
    def tune_free_wavenumber(self, maxiter: int, rtol: float) -> 'Lamb':
        """Refine kx by solving the characteristic equation for a free plate
        
        Parameters:
            maxiter: Maximum number of iterations to run
            rtol: Stopping tolerance
        """
        ...
    
    @abstractmethod
    def tune_immersed_wavenumber(self, maxiter: int, rtol: float) -> 'Lamb':
        """Refine kx by solving the characteristic equation for a fluid-immersed plate
        
        Parameters:
            maxiter: Maximum number of iterations to run
            rtol: Stopping tolerance
        """
        ...
    
    def get_sampled_wavefield(self, samples: int = 100) -> 'SampledWavefield':
        """Get the wavefields of this wave explicitly, sampled along the cross-section
        
        Parameters:
            samples: Number of samples to use
        """
        return SampledWavefield(wave=self, samples=samples)
    
    def get_homogeneous_attenuation(self) -> float:
        """Get the attenuation assuming that the outgoing wave is homogeneous"""
        if np.imag(self.kx) == 0.:   # type: ignore
            wave = self
        else:
            wave = copy(self)
            wave.update_kx(np.real(wave.kx))   # type: ignore
        
        wavefield = wave.get_sampled_wavefield()
        Px = wavefield.get_power_flow()
        Irad_tot = wavefield.get_total_radiated_intensity()
        return np.abs( Irad_tot / (2 * Px) )
    
    def _characteristic_radiation_eq(self,
                                     kxi: Optional[float] = None,
                                     R: Optional[complex] = None,
                                     wavefield_samples: int = 300) -> float:
        """Calculates the characteristic radiation equation given the specified attenuation. NOTE: Alters self!"""
        if kxi is None:
            kxi = self.kx.imag
        else:
            self.update_kx(self.kx.real + 1j*kxi, R=R)
        wavefield = self.get_sampled_wavefield(samples=wavefield_samples)
        Px = wavefield.get_power_flow()
        Irad_tot = wavefield.get_total_radiated_intensity()
        return np.abs(kxi) - Irad_tot / (2 * Px)   # Should work for i(wt-kx) and i(kx-wt) conventions

    def get_boundary_stress_mismatch(self, free_plate: Optional[bool] = None) -> float:
        """Calculate the mismatch between the correct boundary stress and the current boundary stress
        
        Parameters:
            free_plate: Treat the plate as free regardless of whether it is an ImmersedPlate object
        """
        oxy_target = 0
        if free_plate or not isinstance(self.plate, ImmersedPlate):
            oyy_top_target, oyy_bot_target = 0, 0
        else:
            fluid_top, fluid_bot = self.plate.top_material, self.plate.bottom_material
            assert isinstance(fluid_top, FluidMaterial) and isinstance(fluid_bot, FluidMaterial)
            oyy_top_target = - self.p(y=self.h)
            oyy_bot_target = - self.p(y=-self.h)
        
        oxy_top = self.sigma_xy(y=np.array([self.h]))
        oxy_bot = self.sigma_xy(y=np.array([-self.h]))
        oyy_top = self.sigma_yy(y=np.array([self.h]))
        oyy_bot = self.sigma_yy(y=np.array([-self.h]))
        
        return 0.25*(np.abs(oxy_top-oxy_target) + np.abs(oyy_top-oyy_top_target) +
                     np.abs(oxy_bot-oxy_target) + np.abs(oyy_bot-oyy_bot_target))

    def get_inhomogeneous_attenuation(self, inplace: bool = False, optimize_R: bool = False, keep_R: bool = False,
                                      maxiter: int = 2000, rtol: float = 1e-12, verbose=False) -> float:
        """Get the attenuation assuming that the outgoing wave is inhomogeneous, which requires iteration
        
        Parameters:
            inplace: Whether to change self (True) or to change a temporary copy of self (False)
            optimize_R: Whether to also optimize for the P-to-S ratio
            keep_R: If True, keep the current value of R instead of recalculating for free plate
            maxiter: Maximum number of iterations to run
            rtol: Stopping tolerance
            verbose: Whether to spit out low-level optimisation information when optimize_R is True
        """
        if inplace:
            wave = self
        else:
            wave = copy(self)
        
        if optimize_R:
            if verbose: print(f'Optimising for f={wave.f}...')
            scales = np.array([wave.kx.imag, wave.R.real, wave.R.imag])
            start_inputs = np.array([1, 1, 1])
            
            def cost(inputs: Sequence[float]) -> Sequence[float]:
                kxi, Rr, Ri = inputs * scales
                R = Rr + 1j*Ri
                char_eq_cost = wave._characteristic_radiation_eq(kxi=kxi, R=R)
                oxy_cost = 1e-6 * wave.sigma_xy(y=wave.h)   # o_xy(±h) = 0
                oyy_cost = 1e-6 * (wave.sigma_yy(y=wave.h) + wave.p(y=wave.h))   # o_yy(±h) = -p(±h)
                if verbose:
                    print(f'Cost on kxi={kxi}, R={R}')
                    print(f'\tchar_eq_cost = {char_eq_cost}')
                    print(f'\toxy_cost = {oxy_cost}')
                    print(f'\toyy_cost = {oxy_cost}')
                #return char_eq_cost, oxy_cost.real, oxy_cost.imag, oyy_cost.real, oyy_cost.imag
                return oxy_cost.real, oxy_cost.imag, oyy_cost.real, oyy_cost.imag   # meeting BCs is sufficient!
            
            opt.root(cost, start_inputs, method='lm', options={'maxiter': maxiter})

        else:
            R = wave.R if keep_R else None
            
            def cost(kxi):
                return wave._characteristic_radiation_eq(kxi, R)
            
            opt.newton(cost, self.kx.imag, maxiter=maxiter, rtol=rtol)  # type: ignore
        
        return np.abs(np.imag(wave.kx))   # type: ignore

    def tune_immersed_wavenumber_and_R(self, maxiter: int = 2000, rtol: float = 1e-12, verbose=False) -> 'Lamb':
        """Refine kx and R by matching the BCs on both sides of the plate
        
        Unlike refining kx via the characteristic equations, this also gives us R, letting us calculate Lamb wavefields
        
        Parameters:
            maxiter: Maximum number of iterations to run
            rtol: Stopping tolerance
            verbose: Whether to spit out low-level optimisation information
        """
        if verbose: print(f'Optimising for f={self.f}...')
        scales = np.array([self.kx.real, self.kx.imag, self.R.real, self.R.imag])
        start_inputs = np.array([1, 1, 1, 1])
        
        def cost(inputs: Sequence[float]) -> Sequence[float]:
            kxr, kxi, Rr, Ri = inputs * scales
            kx = kxr + 1j * kxi
            R = Rr + 1j * Ri
            self.update_kx(kx=kx, R=R)
            oxy_cost = 1e-6 * self.sigma_xy(y=self.h)   # o_xy(±h) = 0
            oyy_cost = 1e-6 * (self.sigma_yy(y=self.h) + self.p(y=self.h))  # o_yy(±h) = -p(±h)
            if verbose:
                print(f'Cost on kx={kx}, R={R}')
                print(f'\toxy_cost = {oxy_cost}')
                print(f'\toyy_cost = {oxy_cost}')
            return oxy_cost.real, oxy_cost.imag, oyy_cost.real, oyy_cost.imag  # meeting BCs is sufficient?
        
        opt.root(cost, start_inputs, method='lm', options={'maxiter': maxiter})
        return self
    
    def tune_immersed_R(self, maxiter: int = 2000, tol: float = 1e-12, verbose=False) -> 'Lamb':
        """Assuming kx to be exact, refine R by matching the BCs on both sides of the plate
        
        Parameters:
            maxiter: Maximum number of iterations to run
            tol: Stopping tolerance
            verbose: Whether to spit out low-level optimisation information
        """
        if verbose: print(f'Optimising for f={self.f}...')
        scales = np.array([self.R.real, self.R.imag])
        start_inputs = np.array([1, 1])

        def cost(inputs: Sequence[float]) -> Sequence[float]:
            Rr, Ri = inputs * scales
            R = Rr + 1j * Ri
            self.update_R(R)
            oxy_cost = 1e-6 * self.sigma_xy(y=self.h)  # o_xy(±h) = 0
            oyy_cost = 1e-6 * (self.sigma_yy(y=self.h) + self.p(y=self.h))  # o_yy(±h) = -p(±h)
            if verbose:
                print(f'Cost on R={R}')
                print(f'\toxy_cost = {oxy_cost}')
                print(f'\toyy_cost = {oxy_cost}')
            return oxy_cost.real, oxy_cost.imag, oyy_cost.real, oyy_cost.imag

        opt.root(cost, start_inputs, method='lm', tol=tol, options={'maxiter': maxiter})
        return self


class Lamb_A(Lamb):
    """Antisymmetric Lamb wave
    
    Parameters:
        plate: The plate in which the wavefield exists, represented as a Plate object
        kx: The wavenumber along the plate (can be tuned if it's not totally accurate to begin with)
        f: The frequency of the wave (if w is not specified)
        w: The angular frequency of the wave (if f is not specified)
        kx_wt_convention: Whether to use the i(kx-wt) convention instead of the i(wt-kx) convention
    """
    def __init__(self, plate: Plate,
                 kx: float, f: Optional[float] = None, w: Optional[float] = None,
                 kx_wt_convention: bool = True) -> None:
        super().__init__(plate, kx, f, w, kx_wt_convention)
        self.update_R()
    
    def update_R(self, R: Optional[complex] = None) -> None:
        if R is not None:
            self.R = R
        else:
            self.R = (self.kx ** 2 - self.gs ** 2) * np.sin(self.gp * self.h) \
                     / (2 * self.kx * self.gs * np.sin(self.gs * self.h) + 1e-10)
    
    def ux(self, x: coordinates = 0, y: coordinates = 0, t: coordinates = 0):
        x, y, t = self._initialize_coordinate_arrays(x, y, t)
        
        if self.kx_wt_convention:
            common = - self.K * np.exp(1j*(self.kx*x - self.w*t))
        else:
            common = self.K * np.exp(1j*(self.w*t - self.kx*x))
        first = self.kx
        second = -self.gs*self.R
        return np.squeeze(common*(first*np.sin(self.gp*y) + second*np.sin(self.gs*y)))

    def uy(self, x: coordinates = 0, y: coordinates = 0, t: coordinates = 0):
        x, y, t = self._initialize_coordinate_arrays(x, y, t)
        
        if self.kx_wt_convention:
            common = 1j*self.K * np.exp(1j*(self.kx*x - self.w*t))
        else:
            common = 1j*self.K * np.exp(1j*(self.w*t - self.kx*x))
        first = self.gp
        second = self.kx*self.R
        return np.squeeze(common*(first*np.cos(self.gp*y) + second*np.cos(self.gs*y)))

    def N(self):
        inbrackets = self.gp*np.cos(self.gp*self.h) + self.kx*self.R*np.cos(self.gs*self.h)
        return -self.w*inbrackets

    def sigma_xx(self, x: coordinates = 0, y: coordinates = 0, t: coordinates = 0):
        x, y, t = self._initialize_coordinate_arrays(x, y, t)
        
        if self.kx_wt_convention:
            common = self.K*1j*self.G * np.exp(1j*(self.kx*x - self.w*t))
        else:
            common = self.K*1j*self.G * np.exp(1j*(self.w*t - self.kx*x))
        first = 2*self.gp**2 - self.kx**2 - self.gs**2
        second = 2*self.R*self.kx*self.gs
        return np.squeeze(common*(first*np.sin(self.gp*y) + second*np.sin(self.gs*y)))

    def sigma_xy(self, x: coordinates = 0, y: coordinates = 0, t: coordinates = 0):
        x, y, t = self._initialize_coordinate_arrays(x, y, t)
        
        if self.kx_wt_convention:
            common = - self.K * np.exp(1j*(self.kx*x - self.w*t)) * self.G
        else:
            common = self.K * np.exp(1j*(self.w*t - self.kx*x)) * self.G
        first = 2*self.kx*self.gp
        second = self.R*(self.kx**2 - self.gs**2)
        return np.squeeze(common*(first*np.cos(self.gp*y) + second*np.cos(self.gs*y)))

    def sigma_yy(self, x: coordinates = 0, y: coordinates = 0, t: coordinates = 0):
        x, y, t = self._initialize_coordinate_arrays(x, y, t)
        
        if self.kx_wt_convention:
            common = 1j*self.K * np.exp(1j*(self.kx*x - self.w*t)) * self.G
        else:
            common = 1j*self.K * np.exp(1j*(self.w*t - self.kx*x)) * self.G
        first = (self.kx**2 - self.gs**2)
        second = -2*self.R*self.kx*self.gs
        return np.squeeze(common*(first*np.sin(self.gp*y) + second*np.sin(self.gs*y)))
    
    def tune_free_wavenumber(self, maxiter: int = 2000, rtol: float = 1e-12) -> 'Lamb_A':
        self.kx = opt.newton(lambda kx: Ab(self.w, self.d, kx, self.cP, self.cS), x0=self.kx,
                             maxiter=maxiter, rtol=rtol)
        
        self.update_g()
        self.update_R()
        return self   # to support method chaining
    
    def tune_immersed_wavenumber(self, maxiter: int = 2000, rtol: float = 1e-12) -> 'Lamb_A':
        assert isinstance(self.plate, ImmersedPlate)
        assert isinstance(self.plate.top_material, FluidMaterial)
        assert self.plate.top_material == self.plate.bottom_material
        fluid = self.plate.top_material
        char_eq_kx = lambda kx: Ab_leaky(self.w, kx, self.cS, self.cP, self.rho, fluid.rho, fluid.cP, self.plate.d/2)
        
        # Ab_leaky uses the i(kx-wt) convention and not i(wt-kx), which may need to be compensated for
        if not self.kx_wt_convention:
            self.kx = np.conj(self.kx)   # Switch to i(kx-wt)
        self.kx = opt.newton(char_eq_kx, x0=self.kx, maxiter=maxiter, rtol=rtol)  # type: ignore
        if not self.kx_wt_convention:
            self.kx = np.conj(self.kx)   # Switch back to i(wt-kx)
        
        self.update_g()
        self.update_R()   # TODO: We need to calculate a slightly different R for leaky waves
        return self   # to support method chaining
    
    def Px(self, n=100):
        y_samples = np.linspace(-self.h, self.h, n)
        vx = self.vx(y=y_samples)
        vy = self.vy(y=y_samples)
        oxx = self.sigma_xx(y=y_samples)
        oxy = self.sigma_xy(y=y_samples)
        Pxx = integrate.simps(np.real(vx*np.conj(oxx)), x=y_samples)
        Pxy = integrate.simps(np.real(vy*np.conj(oxy)), x=y_samples)
        Px = -0.5*(Pxx+Pxy)
        return Px


class Lamb_S(Lamb):
    """Symmetric Lamb wave

    Parameters:
        plate: The plate in which the wavefield exists, represented as a Plate object
        kx: The wavenumber along the plate (can be tuned if it's not totally accurate to begin with)
        f: The frequency of the wave (if w is not specified)
        w: The angular frequency of the wave (if f is not specified)
        kx_wt_convention: Whether to use the i(kx-wt) convention instead of the i(wt-kx) convention
    """
    def __init__(self, plate: Plate,
                 kx: float, f: Optional[float] = None, w: Optional[float] = None,
                 kx_wt_convention: bool = True) -> None:
        super().__init__(plate, kx, f, w, kx_wt_convention)
        self.update_R()
    
    def update_R(self, R: Optional[complex] = None) -> None:
        if R is not None:
            self.R = R
        else:
            self.R = (self.kx ** 2 - self.gs ** 2) * np.cos(self.gp * self.h) \
                     / (2 * self.kx * self.gs * np.cos(self.gs * self.h) + 1e-10)
        
    def ux(self, x: coordinates = 0, y: coordinates = 0, t: coordinates = 0):
        x, y, t = self._initialize_coordinate_arrays(x, y, t)

        if self.kx_wt_convention:
            logging.warning('i(kx-wt) convention has not been tested for symmetric Lamb waves')
            common = - self.K * np.exp(1j*(self.kx*x - self.w*t))
        else:
            common = self.K * np.exp(1j*(self.w*t - self.kx*x))
        first = self.kx 
        second = -self.gs*self.R
        return np.squeeze(common*(first*np.cos(self.gp*y) + second*np.cos(self.gs*y)))

    def uy(self, x: coordinates = 0, y: coordinates = 0, t: coordinates = 0):
        x, y, t = self._initialize_coordinate_arrays(x, y, t)

        if self.kx_wt_convention:
            logging.warning('i(kx-wt) convention has not been tested for symmetric Lamb waves')
            common = -1j*self.K * np.exp(1j*(self.kx*x - self.w*t))
        else:
            common = -1j*self.K * np.exp(1j*(self.w*t - self.kx*x))
        first = self.gp
        second = self.kx*self.R
        return np.squeeze(common*(first*np.sin(self.gp*y) + second*np.sin(self.gs*y)))

    def N(self):
        inbrackets = 0   # self.gp*np.cos(self.gp*self.h) + self.kx*self.R*np.cos(self.gs*self.h)
        return -self.w*inbrackets

    def sigma_xx(self, x: coordinates = 0, y: coordinates = 0, t: coordinates = 0):
        x, y, t = self._initialize_coordinate_arrays(x, y, t)
        
        if self.kx_wt_convention:
            logging.warning('i(kx-wt) convention has not been tested for symmetric Lamb waves')
            common = self.K*1j*self.G * np.exp(1j*(self.kx*x - self.w*t))
        else:
            common = self.K*1j*self.G * np.exp(1j*(self.w*t - self.kx*x))
        first = 2*self.gp**2 - self.kx**2 - self.gs**2
        second = 2*self.R*self.kx*self.gs
        return np.squeeze(common*(first*np.cos(self.gp*y) + second*np.cos(self.gs*y)))

    def sigma_xy(self, x: coordinates = 0, y: coordinates = 0, t: coordinates = 0):
        x, y, t = self._initialize_coordinate_arrays(x, y, t)
        
        if self.kx_wt_convention:
            logging.warning('i(kx-wt) convention has not been tested for symmetric Lamb waves')
            common = self.K*self.G * np.exp(1j*(self.kx*x - self.w*t))
        else:
            common = -self.K*self.G * np.exp(1j*(self.w*t - self.kx*x))
        first = 2*self.kx*self.gp
        second = self.R*(self.kx**2 - self.gs**2)
        return np.squeeze(common*(first*np.sin(self.gp*y) + second*np.sin(self.gs*y)))

    def sigma_yy(self, x: coordinates = 0, y: coordinates = 0, t: coordinates = 0):
        x, y, t = self._initialize_coordinate_arrays(x, y, t)
        
        if self.kx_wt_convention:
            logging.warning('i(kx-wt) convention has not been tested for symmetric Lamb waves')
            common = 1j*self.K*self.G * np.exp(1j*(self.w*t - self.kx*x))
        else:
            common = 1j*self.K*self.G * np.exp(1j*(self.w*t - self.kx*x))
        first = (self.kx**2 - self.gs**2)
        second = -2*self.R*self.kx*self.gs
        return np.squeeze(common*(first*np.cos(self.gp*y) + second*np.cos(self.gs*y)))

    def tune_free_wavenumber(self, maxiter: int = 2000, rtol: float = 1e-12) -> 'Lamb_S':
        self.kx = opt.newton(lambda kx: Sb(self.w, self.d, kx, self.cP, self.cS), x0=self.kx,   # type: ignore
                             maxiter=maxiter, rtol=rtol)
        return self   # to support method chaining
    
    def tune_immersed_wavenumber(self, maxiter: int, rtol: float) -> 'Lamb_S':
        raise NotImplementedError()
    
    def Px(self, n=100):
        y_samples = np.linspace(-self.h, self.h, n)
        vx = self.vx(y=y_samples)
        vy = self.vy(y=y_samples)
        oxx = self.sigma_xx(y=y_samples)
        oxy = self.sigma_xy(y=y_samples)
        Pxx = integrate.simps(np.real(vx*np.conj(oxx)), x=y_samples)
        Pxy = integrate.simps(np.real(vy*np.conj(oxy)), x=y_samples)
        Px = -0.5*(Pxx+Pxy)
        return Px


class SampledWavefield:
    """Get sampled velocity and stress wavefields throughout plate cross-section"""
    
    def __init__(self, wave: Lamb, samples: int = 300) -> None:
        # Get sampled y coordinates along the plate cross-section
        self.wave = wave
        self.y = np.linspace(-wave.d / 2, wave.d / 2, samples)
        
        # Get velocity and stress components at sampled y coordinates
        self.vx = wave.vx(y=self.y)
        self.vy = wave.vy(y=self.y)
        self.oxx = wave.sigma_xx(y=self.y)
        self.oxy = wave.sigma_xy(y=self.y)
        self.oyy = wave.sigma_yy(y=self.y)
    
    def get_power_flow(self) -> float:
        """Calculate power flow from the sampled wavefield"""
        y, vx, vy, oxx, oxy = self.y, self.vx, self.vy, self.oxx, self.oxy
        
        Pxx = integrate.simps(np.real(vx * np.conj(oxx)), x=y)  # power flow first component
        Pxy = integrate.simps(np.real(vy * np.conj(oxy)), x=y)  # power flow second component
        Px = -0.5 * (Pxx + Pxy)  # power flow in x-direction
        return Px
    
    def get_single_radiated_intensity(self, material: FluidMaterial, vy: float) -> float:
        """Calculate the radiated intensity from one side"""
        k = self.wave.w / material.cP
        ky_k = sm.sqrt(k ** 2 - self.wave.kx ** 2) / k
        return np.real(ky_k) / np.abs(ky_k) ** 2 * material.Z * np.abs(vy) ** 2 / 2
    
    def get_total_radiated_intensity(self) -> float:
        """Calculate the total radiated intensity from both sides"""
        assert isinstance(self.wave.plate, ImmersedPlate)
        
        top_material = self.wave.plate.top_material
        assert isinstance(top_material, FluidMaterial)
        Irad_top = self.get_single_radiated_intensity(top_material, self.vy[-1])
        
        if self.wave.plate.bottom_material == top_material:
            return 2 * Irad_top
        
        bottom_material = self.wave.plate.bottom_material
        assert isinstance(bottom_material, FluidMaterial)
        Irad_bottom = self.get_single_radiated_intensity(bottom_material, self.vy[0])
        
        return Irad_top + Irad_bottom
    
    def get_radiation_factor(self) -> float:
        """Calculate the radiation factor C"""
        assert isinstance(self.wave.plate, ImmersedPlate)
        material = self.wave.plate.top_material
        assert self.wave.plate.bottom_material == material
        assert np.abs(self.vy[0]) == np.abs(self.vy[-1])
        
        vy = np.abs(self.vy[0])
        k = self.wave.w / material.cP
        z = material.Z
        Px = self.get_power_flow()
        
        return z * vy ** 2 / (4 * k * Px)
