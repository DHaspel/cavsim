#! /opt/conda/bin/python3
""" Base pipe class containing shared pipe properties """

# Copyright 2019 FAU-iPAT (http://ipat.uni-erlangen.de/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from ..base.components.component import Component


class BasePipe(Component):
    """
    Basic pipe class with shared properties for all pipes
    """

    def __init__(
            self,
            diameter: float,
            length: float,
            wall_thickness: float,
            bulk_modulus: float,
            roughness: float
    ) -> None:
        """
        Initialization of the class

        :param diameter: Diameter of the fluid volume [m]
        :param length: Length of the pipe [m]
        :param wall_thickness: Thickness of the wall [m]
        :param bulk_modulus: Bulk modulus of wall material [Pa]
        :param roughness: Roughness of the wall [m]
        :raises TypeError: Wrong type of at least one parameter
        """
        super(BasePipe, self).__init__()
        if not isinstance(diameter, (int, float)):
            raise TypeError('Wrong type for parameter diameter ({} != {})'.format(type(diameter), float))
        if not isinstance(length, (int, float)):
            raise TypeError('Wrong type for parameter length ({} != {})'.format(type(length), float))
        if not isinstance(wall_thickness, (int, float)):
            raise TypeError('Wrong type for parameter wall_thickness ({} != {})'.format(type(wall_thickness), float))
        if not isinstance(bulk_modulus, (int, float)):
            raise TypeError('Wrong type for parameter bulk_modulus ({} != {})'.format(type(bulk_modulus), float))
        if not isinstance(roughness, (int, float)):
            raise TypeError('Wrong type for parameter roughness ({} != {})'.format(type(roughness), float))
        self._diameter = diameter
        self._length = length
        self._wall_thickness = wall_thickness
        self._bulk_modulus = bulk_modulus
        self._roughness = roughness

    @property
    def diameter(self) -> float:
        """
        Diameter property of the pipe

        :return: Diameter of the pipe [m]
        """
        return self._diameter

    @diameter.setter
    def diameter(self, diameter: float) -> None:
        """
        Setter of the diameter property of the pipe

        :param diameter: New diameter of the pipe [m]
        :raises TypeError: Assigned value is not a float value
        """
        if not isinstance(diameter, (int, float)):
            raise TypeError('Wrong type of assigned value ({} != {})'.format(type(diameter), float))
        self._diameter = diameter

    @property
    def length(self) -> float:
        """
        Length property of the pipe

        :return: Length of the pipe [m]
        """
        return self._length

    @length.setter
    def length(self, length: float) -> None:
        """
        Setter of the length property of the pipe

        :param length: New length of the pipe [m]
        :raises TypeError: Assigned value is not a float value
        """
        if not isinstance(length, (int, float)):
            raise TypeError('Wrong type of assigned value ({} != {})'.format(type(length), float))
        self._length = length

    @property
    def area(self) -> float:
        """
        Area property of the pipe

        :return: Calculated area of the pipe [m²]
        """
        return np.pi * ((self.diameter / 2.0)**2)

    @property
    def volume(self) -> float:
        """
        Volume property of the fluid within the pipe

        :return: Fluid volume within the pipe [m³]
        """
        return self.area * self.length

    @property
    def wall_thickness(self) -> float:
        """
        Wall thickness property of the pipe

        :return: Wall thickness of the pipe [m]
        """
        return self._wall_thickness

    @wall_thickness.setter
    def wall_thickness(self, wall_thickness: float) -> None:
        """
        Setter of the wall thickness property of the pipe

        :param wall_thickness: New wall thickness of the pipe [m]
        :raises TypeError: Assigned value is not a float value
        """
        if not isinstance(wall_thickness, (int, float)):
            raise TypeError('Wrong type of assigned value ({} != {})'.format(type(wall_thickness), float))
        self._wall_thickness = wall_thickness

    @property
    def bulk_modulus(self) -> float:
        """
        Bulk modulus property of the pipe

        :return: Bulk modulus of the pipe [Pa]
        """
        return self._bulk_modulus

    @bulk_modulus.setter
    def bulk_modulus(self, bulk_modulus: float) -> None:
        """
        Setter of the bulk modulus property of the pipe

        :param bulk_modulus: New bulk modulus of the pipe [Pa]
        :raises TypeError: Assigned value is not a float value
        """
        if not isinstance(bulk_modulus, (int, float)):
            raise TypeError('Wrong type of assigned value ({} != {})'.format(type(bulk_modulus), float))
        self._bulk_modulus = bulk_modulus

    @property
    def roughness(self) -> float:
        """
        Roughness property of the pipe

        :return: Roughness of the pipe [m]
        """
        return self._roughness

    @roughness.setter
    def roughness(self, roughness: float) -> None:
        """
        Setter of the roughness property of the pipe

        :param roughness: New roughness of the pipe [m]
        :raises TypeError: Assigned value is not a float value
        """
        if not isinstance(roughness, (int, float)):
            raise TypeError('Wrong type of assigned value ({} != {})'.format(type(roughness), float))
        self._roughness = roughness

    @property
    def norm_speed_of_sound(self) -> float:
        """
        Modified speed of sound property within the pipe under normal conditions

        :return: Modified speed of sound in the pipe [m/s]
        """
        sos = self.fluid.norm_speed_of_sound
        dims = self.diameter / self.wall_thickness
        compress = self.fluid.norm_compressibility / self.bulk_modulus
        return sos / np.sqrt(1.0 + (dims * compress))

    def speed_of_sound(self, pressure: float = None, temperature: float = None) -> float:
        """
        Modified speed of sound within the pipe under given conditions

        :param pressure: Pressure to get modified speed of sound for [Pa]
        :param temperature: Temperature to get modified speed of sound for [K]
        :return: Modified speed of sound within the pipe [m/s]
        """
        sos = self.fluid.speed_of_sound(pressure=pressure, temperature=temperature)
        dims = self.diameter / self.wall_thickness
        compress = self.fluid.compressibility(temperature=temperature) / self.bulk_modulus
        return sos / np.sqrt(1.0 + (dims * compress))
