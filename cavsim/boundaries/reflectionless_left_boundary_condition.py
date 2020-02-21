#! /opt/conda/bin/python3
""" Pipe class implementing a left boundary with given velocity """

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


class LeftBoundaryReflectionFree(BaseBoundary):
    """
    Pipe class implementing the pipe simulation calculations
    """

    def __init__(
            self,
            velocity_p,
            pressure_p,
    ) -> None:
        """
        Initialization of the class

        :param velocity: Given velocity at the boundary [m/s]
        :raises TypeError: Wrong type of at least one parameter
        :raises ValueError: Value of at least one parameter out of bounds
        """
        super(LeftBoundaryReflectionFree, self).__init__()
        if not callable(velocity_p) and not isinstance(velocity_p, (int, float)):
            raise TypeError('Wrong type for parameter velocity ({} != {})'.format(type(velocity), float))
        self._pressure_p = pressure_p
        self._velocity_p = velocity_p
        # Register internal fields
        self._pressure: np.ndarray = self.field_create('pressure', 3)
        self._velocity: np.ndarray = self.field_create('velocity', 3)
        self._friction: np.ndarray = self.field_create('friction', 3)
        self._sos: np.ndarray = self.field_create('speed_of_sound', 3)
        # Create the right connector
        self._right: Connector = Connector(self, [
            ImportChannel(Measure.deltaX, False),
            ExportChannel(Measure.boundaryPoint, lambda: True),
            ImportChannel(Measure.pressureLast, False),
            ExportChannel(Measure.pressureCurrent, lambda: self._pressure[0, -2]),
            ExportChannel(Measure.pressureLast, lambda: self._pressure[1, -2]),
            ImportChannel(Measure.velocityMinusLast, False),
            ExportChannel(Measure.velocityPlusCurrent, lambda: self._velocity[0, -2]),
            ExportChannel(Measure.velocityPlusLast, lambda: self._velocity[1, -2]),
            ImportChannel(Measure.frictionLast, False),
            ImportChannel(Measure.BPspeedOfSoundLast, False),
        ])

    @property
    def right(self) -> Connector:
        """
        Right connector property

        :return: Right sided connector of the pipe
        """
        return self._right

    @property
    def pressure_p(self):

        return self._pressure_p

    @property
    def velocity_p(self):

        return self._velocity_p

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
        self.field('velocity')[:, :] = np.ones(self.field('velocity').shape)[:, :] * self.velocity_p
        self.field('pressure')[:, :] = self.fluid.initial_pressure * np.ones(self.field('pressure').shape)[:, :]
        self.field('speed_of_sound')[:, :] = np.zeros(self.field('speed_of_sound').shape)[:, :] * self.fluid.speed_of_sound(self.pressure_p)
        self.field('friction')[:, :] = np.zeros(self.field('friction').shape)[:, :]


    def prepare_next_timestep(self, delta_t: float, next_total_time: float) -> None:
        """
        Prepare the internal state for the next timestep to be calculated

        :param delta_t: Timestep width for the next timestep
        :param next_total_time: Total simulation time at the end of the next timestep
        """
        # Shift all internal fields
        self.fields_move()
        # Set fixed pressure value

    def exchange_last_boundaries(self) -> None:
        """
        Exchange boundary values from previous time steps
        """
        # Exchange previous values with the right boundary
        self._pressure[1, -1] = self.right.value(Measure.pressureLast)
        self._velocity[1, -1] = -self.right.value(Measure.velocityMinusLast)
        self._friction[1, -1] = self.right.value(Measure.frictionLast)
        self._sos[1, -1] = self.right.value(Measure.BPspeedOfSoundLast)

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
        # Get the input fields
        pressure_b = self._pressure[1, -1]
        velocity = self._velocity[1, 0]
        pressure = self._pressure[1, 0]
        velocity_b = self._velocity[1, 1]
        friction_b = self._friction[1, 1]
        # Calculate fluid properties
        density = self.fluid.density(pressure=pressure_b, temperature=None)
        speed_of_sound = self._sos[1, -1]
        # Perform actual calculation
        self._pressure[0, 0] = 1 / 2.0*(pressure - pressure_b + density * speed_of_sound * (velocity - velocity_b)
                                  # todo: height terms
                                    )
        self._velocity[0, 0] = (1 / 2.0 * (1 / (density * speed_of_sound)
                                           * (pressure - pressure_b)
                                           + velocity - velocity_b
                                           - 2.0 * friction_b * self._delta_t))

        return False