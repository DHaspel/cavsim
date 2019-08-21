#! /opt/conda/bin/python3
""" Base fluid class """

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


class BaseFluid:
    """
    Basic fluid class (with only values at normal conditions)
    """

    def __init__(
            self,
            density: float,  # [kg/m³]
            viscosity: float,  # [Pa s]
            compressibility: float,  # [1 / Pa]
            vapor_pressure: float,  # [Pa]
            pressure: float = 101325,  # 1 atm = 101 kPa [Pa]
            temperature: float = 293.15  # 20°C [K]
    ) -> None:
        """
        Initializes the fluid

        Properties are given for the specified conditions (pressure/temperature) where
        their default values are for NIST normal conditions (1 atm, 20°C).

        :param density: Density of the fluid [kg/m³]
        :param viscosity: Dynamic viscosity of the fluid [Pa s]
        :param compressibility: Compressibility of the fluid [1 / Pa]
        :param vapor_pressure: Vapor pressure of the fluid [Pa]
        :param pressure: Pressure the properties are given for (default 1 atm)
        :param temperature: Temperature the properties are given for (default 20°C)
        :raises TypeError: Wrong type of at least one parameter
        """
        if not isinstance(density, (int, float)):
            raise TypeError('Wrong type for parameter density ({} != {})'.format(type(density), float))
        if not isinstance(viscosity, (int, float)):
            raise TypeError('Wrong type for parameter viscosity ({} != {})'.format(type(viscosity), float))
        if not isinstance(compressibility, (int, float)):
            raise TypeError('Wrong type for parameter compressibility ({} != {})'.format(type(compressibility), float))
        if not isinstance(vapor_pressure, (int, float)):
            raise TypeError('Wrong type for parameter vapor_pressure ({} != {})'.format(type(vapor_pressure), float))
        if not isinstance(pressure, (int, float)):
            raise TypeError('Wrong type for parameter pressure ({} != {})'.format(type(pressure), float))
        if not isinstance(temperature, (int, float)):
            raise TypeError('Wrong type for parameter temperature ({} != {})'.format(type(temperature), float))
        self._norm_density: float = density
        self._norm_viscosity: float = viscosity
        self._norm_compressibility: float = compressibility
        self._norm_vapor_pressure: float = vapor_pressure
        self._norm_pressure: float = pressure
        self._norm_temperature: float = temperature

    @property
    def norm_pressure(self) -> float:
        """
        Condition pressure the normal values are given for

        :return: Pressure of normal conditions [Pa]
        """
        return self._norm_pressure

    @property
    def norm_temperature(self) -> float:
        """
        Condition temperature the normal values are given for

        :return: Temperature of normal conditions [K]
        """
        return self._norm_temperature

    @property
    def norm_density(self) -> float:
        """
        Density at the normal conditions

        :return: Density at normal conditions [kg/m³]
        """
        return self._norm_density

    @property
    def norm_viscosity(self) -> float:
        """
        Dynamic Viscosity at the normal conditions

        :return: Dynamic viscosity at normal conditions [Pa s]
        """
        return self._norm_viscosity

    @property
    def norm_compressibility(self) -> float:
        """
        Compressibility at the normal conditions

        :return: Compressibility at normal conditions [1 / Pa]
        """
        return self._norm_compressibility

    @property
    def norm_vapor_pressure(self) -> float:
        """
        Vapor pressure at the normal conditions

        :return: Vapor pressure at normal conditions [Pa]
        """
        return self._norm_vapor_pressure

    # noinspection PyUnusedLocal
    def density(self, pressure: float = None, temperature: float = None) -> float:  # pylint: disable=unused-argument
        """
        Calculate the density under the given conditions

        :param pressure: Pressure to get density for
        :param temperature: Temperature to get density for
        :return: Density under the conditions [kg/m³]
        """
        return self.norm_density

    # noinspection PyUnusedLocal,PyPep8
    def viscosity(self, temperature: float = None, shear_rate: float = None) -> float:  # pylint: disable=unused-argument
        """
        Calculate the dynamic viscosity under the given conditions

        :param temperature: Temperature to get viscosity for
        :param shear_rate: Shear rate to get viscosity for
        :return: Dynamic viscosity under the conditions [Pa s]
        """
        return self.norm_viscosity

    def kinematic_viscosity(self, pressure: float = None, temperature: float = None, shear_rate: float = None) -> float:
        """
        Calculate the kinematic viscosity under the given conditions

        :param pressure: Pressure to get viscosity for
        :param temperature: Temperature to get viscosity for
        :param shear_rate: Shear rate to get viscosity for
        :return: Kinematic viscosity under the conditions [m²/s]
        """
        return self.viscosity(temperature, shear_rate) / self.density(pressure, temperature)

    # noinspection PyUnusedLocal
    def compressibility(self, temperature: float = None) -> float:  # pylint: disable=unused-argument
        """
        Calculate the compressibility under the given conditions

        :param temperature: Temperature to get compressibility for
        :return: Compressibility under the conditions [1 / Pa]
        """
        return self.norm_compressibility

    # noinspection PyUnusedLocal
    def vapor_pressure(self, temperature: float = None) -> float:  # pylint: disable=unused-argument
        """
        Calculate the vapor pressure under the given conditions

        :param temperature: Temperature to get vapor pressure for
        :return: Vapor pressure under the conditions [Pa]
        """
        return self.norm_vapor_pressure
