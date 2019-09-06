#! /opt/conda/bin/python3
""" Pipe class implementing the actual pipe simulation calculations """

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


from typing import Optional
import numpy as np
from .base_pipe import BasePipe
from ..base.connectors.connector import Connector
from ..measure import Measure
from ..base.channels.import_channel import ImportChannel
from ..base.channels.export_channel import ExportChannel


class Pipe(BasePipe):
    """
    Pipe class implementing the pipe simulation calculations
    """

    def __init__(
            self,
            diameter: float,
            length: float,
            wall_thickness: float,
            bulk_modulus: float,
            roughness: float,
            inner_points: int = None,
    ) -> None:
        """
        Initialization of the class

        :param diameter: Diameter of the fluid volume [m]
        :param length: Length of the pipe [m]
        :param wall_thickness: Thickness of the wall [m]
        :param bulk_modulus: Bulk modulus of wall material [Pa]
        :param roughness: Roughness of the wall [m]
        :param inner_points: Minimal number of inner points for discretization
        :raises TypeError: Wrong type of at least one parameter
        :raises ValueError: Value of at least one parameter out of bounds
        """
        super(Pipe, self).__init__(diameter, length, wall_thickness, bulk_modulus, roughness)
        if inner_points is not None and not isinstance(inner_points, int):
            raise TypeError('Wrong type for parameter inner_points ({} != {})'.format(type(inner_points), int))
        if inner_points is not None and inner_points < 3:
            raise ValueError('Number of inner points ({}) needs to greater than 2!'.format(inner_points))
        # Register internal fields
        self._inner_points = inner_points
        self._pressure: np.ndarray = self.field_create('pressure', 3)
        self._velocity: np.ndarray = self.field_create('velocity', 3)
        self.field_create('reynolds', 3)
        self.field_create('darcy_friction_factor', 3)
        self.field_create('friction_steady', 3)
        self._friction: np.ndarray = self.field_create('friction', 3)
        self._sos: np.ndarray = self.field_create('speed_of_sound', 3)
        # Create the left connector
        self._left: Connector = Connector(self, [
            ExportChannel(Measure.deltaX, lambda: self._delta_x),
            ImportChannel(Measure.boundaryPoint, False),
            ExportChannel(Measure.diameter, lambda: self.diameter),
            ExportChannel(Measure.length, lambda: self.length),
            ExportChannel(Measure.area, lambda: self.area),
            ImportChannel(Measure.pressureLast, False),
            ExportChannel(Measure.pressureCurrent, lambda: self._pressure[0, 1]),
            ExportChannel(Measure.pressureLast, lambda: self._pressure[1, 1]),
            ImportChannel(Measure.velocityPlusLast, False),
            ExportChannel(Measure.velocityMinusCurrent, lambda: -self._velocity[0, 1]),
            ExportChannel(Measure.velocityMinusLast, lambda: -self._velocity[1, 1]),
            ExportChannel(Measure.frictionCurrent, lambda: self._friction[0, 1]),
            ExportChannel(Measure.frictionLast, lambda: self._friction[1, 1]),
            ExportChannel(Measure.BPspeedOfSoundCurrent, lambda: self._sos[0, 0]),
            ExportChannel(Measure.BPspeedOfSoundLast, lambda: self._sos[1, 0]),
        ])
        # Create the right connector
        self._right: Connector = Connector(self, [
            ExportChannel(Measure.deltaX, lambda: self._delta_x),
            ImportChannel(Measure.boundaryPoint, False),
            ExportChannel(Measure.diameter, lambda: self.diameter),
            ExportChannel(Measure.length, lambda: self.length),
            ExportChannel(Measure.area, lambda: self.area),
            ImportChannel(Measure.pressureLast, False),
            ExportChannel(Measure.pressureCurrent, lambda: self._pressure[0, -2]),
            ExportChannel(Measure.pressureLast, lambda: self._pressure[1, -2]),
            ImportChannel(Measure.velocityMinusLast, False),
            ExportChannel(Measure.velocityPlusCurrent, lambda: self._velocity[0, -2]),
            ExportChannel(Measure.velocityPlusLast, lambda: self._velocity[1, -2]),
            ExportChannel(Measure.frictionCurrent, lambda: self._friction[0, -2]),
            ExportChannel(Measure.frictionLast, lambda: self._friction[1, -2]),
            ExportChannel(Measure.BPspeedOfSoundCurrent, lambda: self._sos[0, -1]),
            ExportChannel(Measure.BPspeedOfSoundLast, lambda: self._sos[1, -1]),
        ])

    @property
    def left(self) -> Connector:
        """
        Left connector property

        :return: Left sided connector of the pipe
        """
        return self._left

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
        n_min = self._inner_points if self._inner_points is not None else 3
        result = self.length / ((n_min + 1) * self.norm_speed_of_sound)
        return result

    def discretize(self, delta_t: float) -> None:
        """
        Method handling the discretization of the component (for a given timestep width)

        :param delta_t: Timestep width to discretize for
        :raises ValueError: Timestep too large to fit at least 3 inner points
        """
        self._delta_t = delta_t
        nodes = int(np.ceil(self.length / (self.norm_speed_of_sound * delta_t)) - 1)
        if nodes < 3:
            raise ValueError('Timestep to large!')
        self._delta_x = self.length / float(nodes + 1)
        self.fields_resize(nodes + 2)

    def initialize(self) -> None:
        """
        Initialize the internal state of the component (after discretization was called)
        """
        self.field('velocity')[:, :] = np.zeros(self.field('velocity').shape)[:, :]
        self.field('pressure')[:, :] = self.fluid.norm_pressure * np.ones(self.field('pressure').shape)[:, :]
        # Initialize derived properties
        for _ in range(2):
            self._calculate_reynolds()
            self._calculate_friction()
            self._calculate_speed_of_sound()
            self.fields_move()

    def prepare_next_timestep(self, delta_t: float, next_total_time: float) -> None:
        """
        Prepare the internal state for the next timestep to be calculated

        :param delta_t: Timestep width for the next timestep
        :param next_total_time: Total simulation time at the end of the next timestep
        """
        # Shift all internal fields
        self.fields_move()

    def exchange_last_boundaries(self) -> None:
        """
        Exchange the boundary values from the last time steps
        """
        # Exchange previous values with the left boundary
        self._pressure[1, 0] = self.left.value(Measure.pressureLast)
        self._velocity[1, 0] = self.left.value(Measure.velocityPlusLast)
        # Exchange previous values with the right boundary
        self._pressure[1, -1] = self.right.value(Measure.pressureLast)
        self._velocity[1, -1] = -self.right.value(Measure.velocityMinusLast)

    def prepare_next_inner_iteration(self, iteration: int) -> None:
        """
        Method to prepare the internal state for the next inner iteration of the current timestep

        :param iteration: Number of the next inner iteration to prepare for
        """

    def exchange_current_boundaries(self) -> None:
        """
        Exchange boundary values from the current time step
        """

    def calculate_next_inner_iteration(self, iteration: int) -> bool:
        """
        Method to do the calculations of the next inner iteration

        :param iteration: Number of the next inner iteration
        :return: Whether this component needs another inner iteration afterwards
        """
        self._calculate_pressure()
        self._calculate_velocity()
        # Calculate static values
        self._calculate_reynolds()
        self._calculate_friction()
        self._calculate_speed_of_sound()
        return False

    def _calculate_speed_of_sound(self) -> None:
        """
        Calculate the current speed of sound
        """
        pressure = self.field_wide_slice('pressure', 0)
        result = self.speed_of_sound(pressure=pressure, temperature=None)
        self.field_wide_slice('speed_of_sound', 0)[:] = result[:]

    def _calculate_reynolds(self) -> None:
        """
        Calculate the Reynolds number based on the values from the previous time step
        """
        # Get the input fields
        pressure = self.field_wide_slice('pressure', 0)
        velocity = self.field_wide_slice('velocity', 0)
        # Calculate fluid properties
        viscosity = self.fluid.viscosity(temperature=None, shear_rate=None)
        density = self.fluid.density(pressure=pressure, temperature=None)
        # Calculate the reynolds number
        result = (density * np.abs(velocity) * self.diameter) / viscosity
        # Store/return the calculated result
        self.field_wide_slice('reynolds', 0)[:] = result[:]

    def _calculate_darcy_friction_factor(self) -> None:
        """
        Calculates darcy's friction coefficient within the pipe
        """
        # Get the input fields
        reynolds = self.field_wide_slice('reynolds', 0)
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
        self.field_wide_slice('darcy_friction_factor', 0)[:] = result[:]

    def _calculate_friction_steady(self) -> None:
        """
        Calculate the steady friction using darcy's factor
        """
        # Get the input fields
        velocity = self.field_wide_slice('velocity', 0)
        friction_factor = self.field_wide_slice('darcy_friction_factor', 0)
        # Calculate the friction
        result = (friction_factor / (2.0 * self.diameter)) * np.abs(velocity) * velocity
        # Store/return the calculated result
        self.field_wide_slice('friction_steady', 0)[:] = result[:]

    def _calculate_friction(self) -> None:
        """
        Calculate the total friction (steady + unsteady)
        """
        self._calculate_darcy_friction_factor()
        self._calculate_friction_steady()
        # todo: calculate unsteady friction
        friction_steady = self.field_wide_slice('friction_steady', 0)
        result = friction_steady
        self.field_wide_slice('friction', 0)[:] = result[:]

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
        friction_a = self.field_slice('friction', 1, -1)
        friction_b = self.field_slice('friction', 1, +1)
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
        friction_a = self.field_slice('friction', 1, -1)
        friction_b = self.field_slice('friction', 1, +1)
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
