from typing import Optional, Union
import numpy as np
from ..base.connectors.connector import Connector
from ..measure import Measure
from ..base.channels.import_channel import ImportChannel
from ..base.channels.export_channel import ExportChannel

class Valve():

    def __init__(self,
                 outer_diameter: float,
                 inner_diameter: float,
                 fluid_density: float,
                 material_density: float,
                 spring_stiffness: float,
                 initial_spring_force: float,
                 valve_mass: float,
                 spring_mass: float,
                 suction_pressure: float,
                 )-> None:

        self._fluid_density = fluid_density
        self._material_density = material_density
        self._spring_stiffness = spring_stiffness
        self._initial_spring_force = initial_spring_force
        self._valve_mass = valve_mass
        self._spring_mass = spring_mass
        self._buoyancy_force = None
        self._suction_pressure = suction_pressure
        self._suction_force = None
        self._outer_diameter = outer_diameter
        self._inner_diameter = inner_diameter

    @property
    def fluid_density(self):

        return self._fluid_density

    @property
    def material_density(self):

        return self._material_density

    @property
    def spring_stiffness(self):

        return self._spring_stiffness

    @property
    def initial_spring_force(self):

        return self._initial_spring_force

    @property
    def valve_mass(self):

        return self._valve_mass

    @property
    def spring_mass(self):

        return self._spring_mass

    @property
    def suction_pressure(self):

        return self._suction_pressure

    @property
    def outer_diameter(self):

        return self._outer_diameter

    @property
    def inner_diameter(self):

        return self._inner_diameter

    @property
    def valve_area(self):

        return np.pi * np.power(self._outer_diameter, 2) / 4

    @property
    def contact_area(self):

        return np.pi * ((self._outer_diameter / 2)**2 - (self._inner_diameter / 2)**2)

    def calculate_buoyancy_force(self):

        self._buoyancy_force = (((self._material_density - self._fluid_density)
                                 / self._material_density)
                                * (self._valve_mass + self._spring_mass))
        return None

    def calculate_lower_force(self):

        return (self._valve_area - self._contact_area) * self._suction_pressure

    def calculate_upper_force(self):

        return self._valve_area * self._pressure_ar

    def calculate_spring_force(self):

        return self._initial_spring_force + self._spring_stiffness * self._displacement

