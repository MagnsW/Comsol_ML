from .materials import FluidMaterial, ElasticMaterial


class Plate:
    """A solid plate with a thickness d"""
    def __init__(self, d: float, material: ElasticMaterial) -> None:
        self.d = d
        self.material = material


class ImmersedPlate(Plate):
    """Subclass of Plate, for a plate with semi-infinite fluids above and below"""
    def __init__(self,
                 d: float, material: ElasticMaterial,
                 top_material: FluidMaterial, bottom_material: FluidMaterial) -> None:
        super().__init__(d, material)
        self.top_material = top_material
        self.bottom_material = bottom_material
