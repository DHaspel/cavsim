#! /opt/conda/bin/python3
""" Pipe class implementing a right boundary with given velocity """

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


class RightBoundaryVelocity(BaseBoundary):
    """
    Pipe class implementing the pipe simulation calculations
    """

    def __init__(
            self,
            velocity: Union[float, BoundaryFunction],
            initial_pressure: float,
    ) -> None:
        """
        Initialization of the class

        :param velocity: Given velocity at the boundary [m/s]
        :raises TypeError: Wrong type of at least one parameter
        :raises ValueError: Value of at least one parameter out of bounds
        """
        super(RightBoundaryVelocity, self).__init__()
        if not callable(velocity) and not isinstance(velocity, (int, float)):
            raise TypeError('Wrong type for parameter velocity ({} != {})'.format(type(velocity), float))
        # Register internal fields
        self._boundary = velocity
        self._initial_pressure = initial_pressure
        self._pressure: np.ndarray = self.field_create('pressure', 3)
        self._velocity: np.ndarray = self.field_create('velocity', 3)
        self._friction: np.ndarray = self.field_create('friction', 3)
        self._sos: np.ndarray = self.field_create('speed_of_sound', 3)
        self._delta_x_left = 1.0
        self._delta_x_a = 1.0
        self._pressure_a = 0.0
        # Create the left connector
        self._left: Connector = Connector(self, [
            ImportChannel(Measure.deltaX, False),
            ExportChannel(Measure.boundaryPoint, lambda: True),
            ImportChannel(Measure.pressureLast, False),
            ExportChannel(Measure.pressureCurrent, lambda: self._pressure[0, 1]),
            ExportChannel(Measure.pressureLast, lambda: self._pressure[1, 1]),
            ImportChannel(Measure.velocityPlusLast, False),
            ImportChannel(Measure.dxdt, False),
            ExportChannel(Measure.velocityMinusCurrent, lambda: -self._velocity[0, 1]),
            ExportChannel(Measure.velocityMinusLast, lambda: -self._velocity[1, 1]),
            ImportChannel(Measure.frictionLast, False),
            ImportChannel(Measure.BPspeedOfSoundLast, False),
        ])

    @property
    def left(self) -> Connector:
        """
        Left connector property

        :return: Left sided connector of the pipe
        """
        return self._left

    @property
    def initial_pressure(self):
        """

        """
        return self._initial_pressure

    def get_max_delta_t(self) -> Optional[float]:
        """
        Method to return the maximum allowed timestep width for this component

        :return: Maximum allowed timestep width or None if any is suitable
        """

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
        self.field('velocity')[:, :] = np.zeros(self.field('velocity').shape)[:, :]
        if self.initial_pressure is not None:
            self.field('pressure')[:, :] = self.initial_pressure * np.ones(self.field('pressure').shape)[:, :]
            self._pressure_a = self.initial_pressure
        else:
            self.field('pressure')[:, :] = self.fluid.initial_pressure * np.ones(self.field('pressure').shape)[:, :]
            self._pressure_a = self.fluid.initial_pressure
        self.field('friction')[:, :] = np.zeros(self.field('friction').shape)[:, :]
        self.field('speed_of_sound')[:, :] = np.zeros(self.field('friction').shape)[:, :]
        self._delta_x_left = self.left.value(Measure.deltaX)
        self._delta_x_a = self.left.value(Measure.deltaX)

    def prepare_next_timestep(self, delta_t: float, next_total_time: float) -> None:
        """
        Prepare the internal state for the next timestep to be calculated

        :param delta_t: Timestep width for the next timestep
        :param next_total_time: Total simulation time at the end of the next timestep
        """
        # Shift all internal fields
        self.fields_move()
        # Set fixed pressure value
        self._velocity[0, 1] = self._boundary(next_total_time) if callable(self._boundary) else self._boundary

    def exchange_last_boundaries(self) -> None:
        """
        Exchange boundary values from previous time steps
        """
        # Exchange previous values with the left boundary
        self._pressure[1, 0] = self.left.value(Measure.pressureLast)
        self._velocity[1, 0] = self.left.value(Measure.velocityPlusLast)
        self._friction[1, 0] = self.left.value(Measure.frictionLast)
        self._sos[1, 1] = self.left.value(Measure.BPspeedOfSoundLast)

    def prepare_next_inner_iteration(self, iteration: int) -> None:
        """
        Method to prepare the internal state for the next inner iteration of the current timestep

        :param iteration: Number of the next inner iteration to prepare for
        """

    def _calculate_space_interpolation(self) -> None:

        velocity_left = self._velocity[1, 1]
        pressure_left = self._pressure[1, 1]
        velocity_a = self._velocity[1, 0]
        pressure_a = self._pressure[1, 0]
        speed_of_sound_a = self._sos[1, 1]

        self._delta_x_a = (
                (speed_of_sound_a
                 / (self._delta_x_left / self._delta_t
                    + 1.0 / 2.0 * (velocity_left - velocity_a)))
                * self._delta_x_left
        )

        self._pressure_a = (pressure_left
                            - ((pressure_left - pressure_a) / self._delta_x_left)
                            * self._delta_x_a)

        self._velocity_a = (velocity_left
                            - ((velocity_left - velocity_a) / self._delta_x_left)
                            * self._delta_x_a)

    def calculate_next_inner_iteration(self, iteration: int) -> bool:
        """
        Method to do the calculations of the next inner iteration

        :param iteration: Number of the next inner iteration
        :return: Whether this component needs another inner iteration afterwards
        """
        self._calculate_space_interpolation()
        # Get the input fields
        pressure_center = self._pressure[1, 1]
        pressure_a = self._pressure_a
        velocity_p = self._velocity[0, 1]
        velocity_a = self._velocity_a
        friction_a = self._friction[1, 0]
        # Calculate fluid properties
        density = self.fluid.density(pressure=pressure_center, temperature=None)
        speed_of_sound = self._sos[1, 1]
        # Perform actual calculation
        result = pressure_a - density * speed_of_sound * (
            (velocity_p - velocity_a)
            + self._delta_t * friction_a
            # todo: height terms
        )
        self._pressure[0, 1] = result
        return False