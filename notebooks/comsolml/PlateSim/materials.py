import numpy as np


class FluidMaterial:
    """Fluid material
    
    Parameters:
        rho: Mass density in kg/m^3
        c: Speed of sound in m/s
    """
    def __init__(self, rho: float, c: float):
        self.rho = rho
        self.cP = c
        self.Z = self.cP*self.rho


class ElasticMaterial:
    """Elastic isotropic material
    
    Keyword arguments must include one of the following pairs: cS,cP / L,G (Lamé) / E,v. From these, the other linear
    elastic parameters are calculated.
    
    Parameters:
        rho: Mass density in kg/m^3
        **kwargs: Keyword arguments providing a supported pair of linear elastic parameters
    """
    def __init__(self, rho: float, **kwargs):
        self.rho = rho
        if ('cP' in kwargs) and ('cS' in kwargs):
            self.cP = kwargs['cP']
            self.cS = kwargs['cS']
            self.G = self.cS**2 * self.rho
            self.L = self.cP**2 * self.rho - 2*self.G
            self.v = (2*(self.cS/self.cP)**2 - 1) / (2*(self.cS/self.cP)**2 - 2)
            self.E = 2*(1+self.v)*self.G
        elif ('E' in kwargs) and ('v' in kwargs):
            self.E = kwargs['E']
            self.v = kwargs['v']
            self.L = (self.v*self.E)/((1+self.v)*(1-2*self.v))
            self.G = self.E / (2*(1+self.v))
            self.cP = np.sqrt((self.L + 2*self.G)/self.rho)
            self.cS = np.sqrt(self.G/self.rho)
        elif ('G' in kwargs) and ('L' in kwargs):
            self.L = kwargs['L']
            self.G = kwargs['G']
            self.cP = np.sqrt((self.L + 2*self.G)/self.rho)
            self.cS = np.sqrt(self.G/self.rho)
            self.v = (2*(self.cS/self.cP)**2 - 1) / (2*(self.cS/self.cP)**2 - 2)
            self.E = 2*(1+self.v)*self.G
        else:
            print('Need one of the following pairs: cS,cP / L,G (Lamé) / E,v.')


water = ElasticMaterial(rho = 1000, cS = 0.001, cP = 1480)
steel = ElasticMaterial(rho = 7850, cS = 3200, cP = 5900)
light_cement = ElasticMaterial(rho = 1330, cS = 770, cP = 2250)
heavy_cement = ElasticMaterial(rho = 1800, cS = 1850, cP = 3500)
vacuum_water = ElasticMaterial(rho = 0.0001, cS = 0.001, cP = 1480)
airy_water = ElasticMaterial(rho = 1, cS = 0.001, cP = 1480)
brass = ElasticMaterial(rho = 8440, L = 8.72e10, G= 4.10e10 ) 
air = ElasticMaterial(rho = 1, cS = 0.0001, cP = 340)
