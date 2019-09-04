#! /opt/conda/bin/python3
""" Pipe class implementing a left boundary with given pressure """

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


from typing import Optional, Union
import numpy as np
from .base_boundary import BaseBoundary, BoundaryFunction
from ..base.connectors.connector import Connector
from ..measure import Measure
from ..base.channels.import_channel import ImportChannel
from ..base.channels.export_channel import ExportChannel


class LeftBoundaryPressure(BaseBoundary):
    """
    Pipe class implementing the pipe simulation calculations
    """

    def __init__(
            self,
            pressure: Union[float, BoundaryFunction]
    ) -> None:
        """
        Initialization of the class

        :param pressure: Given pressure at the boundary [m]
        :raises TypeError: Wrong type of at least one parameter
        :raises ValueError: Value of at least one parameter out of bounds
        """
        super(LeftBoundaryPressure, self).__init__()
        if not callable(pressure) and not isinstance(pressure, [int,float]):
            raise TypeError('Wrong type for parameter pressure ({} != {})'.format(type(pressure), float))
        # Register internal fields
        # todo: Register fields
        self._pressure: np.ndarray = self.field_create('pressure', 3)
        self._velocity: np.ndarray = self.field_create('velocity', 3)
        """self.field_create('reynolds', 1)
        self.field_create('darcy_friction_factor', 1)
        self.field_create('friction_steady', 1)
        self.field_create('friction', 1)"""
        # Create the right connector
        # todo: Apply correct connectors
        self._right: Connector = Connector(self, [
            ImportChannel(Measure.deltaX, False),
            ExportChannel(Measure.boundaryPoint, lambda: True),
            ImportChannel(Measure.pressureLast, False),
            ExportChannel(Measure.pressureCurrent, lambda: self._pressure[0, -2]),
            ExportChannel(Measure.pressureLast, lambda: self._pressure[1, -2]),
            ImportChannel(Measure.velocityMinusLast, False),
            ExportChannel(Measure.velocityPlusCurrent, lambda: self._velocity[0, -2]),
            ExportChannel(Measure.velocityPlusLast, lambda: self._velocity[1, -2]),
        ])

    @property
    def right(self) -> Connector:
        """
        Right connector property

        :return: Right sided connector of the pipe
        """
        return self._right

    def get_max_delta_t(self) -> Optional[float]:
        """
        Method to return the maximum allowed timestep width for this component

        :return: Maximum allowed timestep width or None if any is suitable
        """
        return None

    def discretize(self, delta_t: float) -> None:
        """
        Method handling the discretization of the component (for a given timestep width)

        :param delta_t: Timestep width to discretize for
        :raises ValueError: Timestep too large to fit at least 3 inner points
        """
        self._delta_t = delta_t
        self.fields_resize(2)

    def initialize(self) -> None:
        """
        Initialize the internal state of the component (after discretization was called)
        """
        # todo: initialize
        self.field('velocity')[:, :] = np.zeros(self.field('velocity').shape)[:, :]
        self.field('pressure')[:, :] = self.fluid.norm_pressure * np.ones(self.field('pressure').shape)[:, :]
        """self.field('reynolds')[:, :] = np.zeros(self.field('reynolds').shape)[:, :]
        self.field('darcy_friction_factor')[:, :] = np.ones(self.field('darcy_friction_factor').shape)[:, :]
        self.field('friction_steady')[:, :] = np.zeros(self.field('friction').shape)[:, :]
        self.field('friction')[:, :] = np.zeros(self.field('friction').shape)[:, :]
        # todo: User defined initialize of the internal states"""

    def prepare_next_timestep(self, delta_t: float, next_total_time: float) -> None:
        """
        Prepare the internal state for the next timestep to be calculated

        :param delta_t: Timestep width for the next timestep
        :param next_total_time: Total simulation time at the end of the next timestep
        """
        # Shift all internal fields
        self.fields_move()
        # Exchange previous values with the right boundary
        self._pressure[1, -1] = self.right.value(Measure.pressureLast)
        self._velocity[1, -1] = -self.right.value(Measure.velocityMinusLast)

    def prepare_next_inner_iteration(self, iteration: int) -> None:
        """
        Method to prepare the internal state for the next inner iteration of the current timestep

        :param iteration: Number of the next inner iteration to prepare for
        """

    def calculate_next_inner_iteration(self, iteration: int) -> bool:
        """
        Method to do the calculations of the next inner iteration

        :param iteration: Number of the next inner iteration
        :return: Whether this component needs another inner iteration afterwards
        """
        # todo: Perform calculation
        return False

    _='''def _calculate_reynolds(self) -> None:
        """
        Calculate the Reynolds number based on the values from the previous time step
        """
        # Get the input fields
        pressure = self.field_wide_slice('pressure', 1)
        velocity = self.field_wide_slice('velocity', 1)
        # Calculate fluid properties
        viscosity = self.fluid.viscosity(temperature=None, shear_rate=None)
        density = self.fluid.density(pressure=pressure, temperature=None)
        # Calculate the reynolds number
        result = (density * velocity * self.diameter) / viscosity
        # Store/return the calculated result
        self.field_wide_slice('reynolds')[:] = result[:]

    def _calculate_darcy_friction_factor(self) -> None:
        """
        Calculates darcy's friction coefficient within the pipe
        """
        # Get the input fields
        reynolds = self.field_wide_slice('reynolds')
        result = np.ones(reynolds.shape)
        # Calculate the friction factor (low Re)
        selector = np.logical_and(reynolds > 0.0, reynolds < 2100.0)
        if np.sum(selector) > 0:
            local_reynolds = reynolds[selector]
            factor = 64.0 / local_reynolds
            result[selector] = factor
        # Calculate the friction factor (high Re)
        selector = (reynolds >= 2100.0)
        if np.sum(selector) > 0:
            local_reynolds = reynolds[selector]
            factor = 10.0 * np.ones(local_reynolds.shape)
            error = np.ones(local_reynolds.shape)
            while np.any(error > 1e-12):
                term1 = self.roughness / (3.7 * self.diameter)
                term2 = 2.51 / (local_reynolds * np.sqrt(factor))
                temp = -2.0 * np.log10(term1 + term2)
                old_factor, factor = factor, np.square(1.0 / temp)
                error = np.abs(factor - old_factor)
            result[selector] = factor
        # Store/return the calculated result
        self.field_wide_slice('darcy_friction_factor')[:] = result[:]

    def _calculate_friction_steady(self) -> None:
        """
        Calculate the steady friction using darcy's factor
        """
        # Get the input fields
        velocity = self.field_wide_slice('velocity', 1)
        friction_factor = self.field_wide_slice('darcy_friction_factor', 0)
        # Calculate the friction
        result = (friction_factor / (2.0 * self.diameter)) * np.abs(velocity)
        # Store/return the calculated result
        self.field_wide_slice('friction_steady')[:] = result[:]

    def _calculate_friction(self) -> None:
        """
        Calculate the total friction (steady + unsteady)
        """
        self._calculate_darcy_friction_factor()
        self._calculate_friction_steady()
        # todo: calculate unsteady friction
        friction_steady = self.field_wide_slice('friction_steady')
        result = friction_steady
        self.field_wide_slice('friction')[:] = result[:]

    def _calculate_pressure(self) -> None:
        """
        Calculate the pressure of the current time step
        """
        # Get the input fields
        pressure_center = self.field_slice('pressure', 1, 0)
        pressure_a = self.field_slice('pressure', 1, -1)
        pressure_b = self.field_slice('pressure', 1, +1)
        velocity_a = self.field_slice('velocity', 1, -1)
        velocity_b = self.field_slice('velocity', 1, +1)
        friction_a = self.field_slice('friction', 0, -1)
        friction_b = self.field_slice('friction', 0, +1)
        # Calculate fluid properties
        speed_of_sound = self.speed_of_sound(pressure=pressure_center, temperature=None)
        density = self.fluid.density(pressure=pressure_center, temperature=None)
        # Calculate the reynolds number
        result = 0.5 * (
            (speed_of_sound * density * (velocity_a - velocity_b))
            + (pressure_a + pressure_b)
            + (self._delta_t * speed_of_sound * density * (friction_b - friction_a))
            # todo: height terms
        )
        # Store/return the calculated result
        self.field_slice('pressure', 0, 0)[:] = result[:]

    def _calculate_velocity(self) -> None:
        """
        Calculate the velocity of the current time step
        """
        # Get the input fields
        pressure_center = self.field_slice('pressure', 1, 0)
        pressure_a = self.field_slice('pressure', 1, -1)
        pressure_b = self.field_slice('pressure', 1, +1)
        velocity_a = self.field_slice('velocity', 1, -1)
        velocity_b = self.field_slice('velocity', 1, +1)
        friction_a = self.field_slice('friction', 0, -1)
        friction_b = self.field_slice('friction', 0, +1)
        # Calculate fluid properties
        speed_of_sound = self.speed_of_sound(pressure=pressure_center, temperature=None)
        density = self.fluid.density(pressure=pressure_center, temperature=None)
        # Calculate the reynolds number
        result = 0.5 * (
            (velocity_a + velocity_b)
            + ((1.0 / (speed_of_sound * density)) * (pressure_a - pressure_b))
            - (self._delta_t * (friction_a + friction_b))
            # todo: height terms
        )
        # Store/return the calculated result
        self.field_slice('velocity', 0, 0)[:] = result[:]
    '''