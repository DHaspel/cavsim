#! /opt/conda/bin/python3
""" Fluid class implementing callback structure for calculations """

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


from typing import Callable, Optional
from .base_fluid import BaseFluid


CallbackDensity = Callable[[BaseFluid, Optional[float], Optional[float]], float]
CallbackViscosity = Callable[[BaseFluid, Optional[float], Optional[float]], float]
CallbackBulkModulus = Callable[[BaseFluid, Optional[float]], float]
CallbackVaporPressure = Callable[[BaseFluid, Optional[float]], float]


class Fluid(BaseFluid):
    """
    Fluid class implementing callback structure for property calculation
    """

    def __init__(
            self,
            density: float,  # [kg/m³]
            viscosity: float,  # [Pa s]
            bulk_modulus: float,  # [Pa]
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
        :param bulk_modulus: Bulk modulus of the fluid [Pa]
        :param vapor_pressure: Vapor pressure of the fluid [Pa]
        :param pressure: Pressure the properties are given for (default 1 atm)
        :param temperature: Temperature the properties are given for (default 20°C)
        """
        super(Fluid, self).__init__(density, viscosity, bulk_modulus, vapor_pressure, pressure, temperature)
        self._density_cb: Optional[CallbackDensity] = None
        self._viscosity_cb: Optional[CallbackViscosity] = None
        self._bulk_modulus_cb: Optional[CallbackBulkModulus] = None
        self._vapor_pressure_cb: Optional[CallbackVaporPressure] = None

    # noinspection PyUnusedLocal
    def _density(self, pressure: float = None, temperature: float = None) -> float:  # pylint: disable=unused-argument
        """
        Override method to calculate the density under the given conditions

        :param pressure: Pressure to get density for
        :param temperature: Temperature to get density for
        :return: Density under the conditions [kg/m³]
        """
        return self.norm_density * self._ones(pressure, temperature)

    # noinspection PyUnusedLocal,PyPep8
    def _viscosity(self, temperature: float = None, shear_rate: float = None) -> float:  # pylint: disable=unused-argument
        """
        Override method to calculate the dynamic viscosity under the given conditions

        :param temperature: Temperature to get viscosity for
        :param shear_rate: Shear rate to get viscosity for
        :return: Dynamic viscosity under the conditions [Pa s]
        """
        return self.norm_viscosity * self._ones(temperature, shear_rate)

    # noinspection PyUnusedLocal
    def _bulk_modulus(self, temperature: float = None) -> float:  # pylint: disable=unused-argument
        """
        Override method to calculate the bulk modulus under the given conditions

        :param temperature: Temperature to get bulk modulus for
        :return: Compressibility under the conditions [Pa]
        """
        return self.norm_bulk_modulus * self._ones(temperature)

    # noinspection PyUnusedLocal
    def _vapor_pressure(self, temperature: float = None) -> float:  # pylint: disable=unused-argument
        """
        Override method to calculate the vapor pressure under the given conditions

        :param temperature: Temperature to get vapor pressure for
        :return: Vapor pressure under the conditions [Pa]
        """
        return self.norm_vapor_pressure * self._ones(temperature)

    def density(self, pressure: float = None, temperature: float = None) -> float:
        """
        Calculate the density under the given conditions

        :param pressure: Pressure to get density for
        :param temperature: Temperature to get density for
        :return: Density under the conditions [kg/m³]
        """
        if callable(self._density_cb):
            return self._density_cb(self, pressure, temperature)  # pylint: disable=not-callable
        return self._density(pressure, temperature)

    def viscosity(self, temperature: float = None, shear_rate: float = None) -> float:
        """
        Calculate the dynamic viscosity under the given conditions

        :param temperature: Temperature to get viscosity for
        :param shear_rate: Shear rate to get viscosity for
        :return: Dynamic viscosity under the conditions [Pa s]
        """
        if callable(self._viscosity_cb):
            return self._viscosity_cb(self, temperature, shear_rate)  # pylint: disable=not-callable
        return self._viscosity(temperature, shear_rate)

    def bulk_modulus(self, temperature: float = None) -> float:
        """
        Calculate the bulk modulus under the given conditions

        :param temperature: Temperature to get bulk modulus for
        :return: Bulk modulus under the conditions [Pa]
        """
        if callable(self._bulk_modulus_cb):
            return self._bulk_modulus_cb(self, temperature)  # pylint: disable=not-callable
        return self._bulk_modulus(temperature)

    def vapor_pressure(self, temperature: float = None) -> float:
        """
        Calculate the vapor pressure under the given conditions

        :param temperature: Temperature to get vapor pressure for
        :return: Vapor pressure under the conditions [Pa]
        """
        if callable(self._vapor_pressure_cb):
            return self._vapor_pressure_cb(self, temperature)  # pylint: disable=not-callable
        return self._vapor_pressure(temperature)

    @property
    def density_cb(self) -> Optional[CallbackDensity]:
        """
        Callback function property for density calculation

        :return: Callback function for density
        """
        return self._density_cb

    @density_cb.setter
    def density_cb(self, function: Optional[CallbackDensity]) -> None:
        """
        Setter method for callback function for density

        :param function: New value for callback function for density
        :raises TypeError: Assigned value is neither None nor a callable
        """
        if function is not None and not callable(function):
            raise TypeError('Wrong type of assigned value ({} != {})'.format(type(function), Callable))
        self._density_cb = function

    @property
    def viscosity_cb(self) -> Optional[CallbackViscosity]:
        """
        Callback function property for viscosity calculation

        :return: Callback function for viscosity
        """
        return self._viscosity_cb

    @viscosity_cb.setter
    def viscosity_cb(self, function: Optional[CallbackViscosity]) -> None:
        """
        Setter method for callback function for viscosity

        :param function: New value for callback function for viscosity
        :raises TypeError: Assigned value is neither None nor a callable
        """
        if function is not None and not callable(function):
            raise TypeError('Wrong type of assigned value ({} != {})'.format(type(function), Callable))
        self._viscosity_cb = function

    @property
    def bulk_modulus_cb(self) -> Optional[CallbackBulkModulus]:
        """
        Callback function property for bulk modulus calculation

        :return: Callback function for bulk modulus
        """
        return self._bulk_modulus_cb

    @bulk_modulus_cb.setter
    def bulk_modulus_cb(self, function: Optional[CallbackBulkModulus]) -> None:
        """
        Setter method for callback function for bulk modulus

        :param function: New value for callback function for bulk modulus
        :raises TypeError: Assigned value is neither None nor a callable
        """
        if function is not None and not callable(function):
            raise TypeError('Wrong type of assigned value ({} != {})'.format(type(function), Callable))
        self._bulk_modulus_cb = function

    @property
    def vapor_pressure_cb(self) -> Optional[CallbackVaporPressure]:
        """
        Callback function property for vapor pressure calculation

        :return: Callback function for vapor pressure
        """
        return self._vapor_pressure_cb

    @vapor_pressure_cb.setter
    def vapor_pressure_cb(self, function: Optional[CallbackVaporPressure]) -> None:
        """
        Setter method for callback function for vapor pressure

        :param function: New value for callback function for vapor pressure
        :raises TypeError: Assigned value is neither None nor a callable
        """
        if function is not None and not callable(function):
            raise TypeError('Wrong type of assigned value ({} != {})'.format(type(function), Callable))
        self._vapor_pressure_cb = function
