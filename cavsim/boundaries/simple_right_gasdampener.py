#! /opt/conda/bin/python3
""" Pipe class implementing a right boundary with given pressure """

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


class RightGasBubbleSimple(BaseBoundary):
    """
    Pipe class implementing the pipe simulation calculations
    """

    def __init__(
            self,
            pressure0: float,
            volume0: float,
            polytropic_exponent: float,
    ) -> None:
        """

        :param pressure0: Starting pressure for the gas bubble
        :param volume0: Starting volume for the gas bubble
        """

        super(RightGasBubbleSimple, self).__init__()
        # Input bubble data
        self._pressure0 = pressure0
        self._volume0 = volume0
        self._polytropic_exponent = polytropic_exponent
        self._area = 1

        # Register internal fields

        self._pressure: np.ndarray = self.field_create('pressure', 3)
        self._velocity: np.ndarray = self.field_create('velocity', 3)
        self._friction: np.ndarray = self.field_create('friction', 3)
        self._sos: np.ndarray = self.field_create('speed_of_sound', 3)
        self._volume: np.ndarray = self.field_create('volume', 3)

        # Create the left connector
        self._left: Connector = Connector(self, [
            ImportChannel(Measure.deltaX, False),
            ExportChannel(Measure.boundaryPoint, lambda: True),
            ImportChannel(Measure.pressureLast, False),
            ExportChannel(Measure.pressureCurrent, lambda: self._pressure[0, 1]),
            ExportChannel(Measure.pressureLast, lambda: self._pressure[1, 1]),
            ImportChannel(Measure.velocityPlusLast, False),
            ExportChannel(Measure.velocityMinusCurrent, lambda: -self._velocity[0, 1]),
            ExportChannel(Measure.velocityMinusLast, lambda: -self._velocity[1, 1]),
            ImportChannel(Measure.frictionLast, False),
            ImportChannel(Measure.BPspeedOfSoundLast, False),
            ImportChannel(Measure.area)
        ])

    @property
    def left(self) -> Connector:
        """
        Left connector property

        :return: Left sided connector of the pipe
        """
        return self._left

    @property
    def pressure0(self):
        """
        Starting pressure property

        :return:
        """
        return self._pressure0

    @property
    def volume0(self):
        """

        :return: Volume of the gas bubble
        """
        return self._volume0

    @property
    def polytropic_exponent(self):
        """

        :return: Polytropic exponent of the compression
        """
        return self._polytropic_exponent

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
        self.field('pressure')[:, :] = self.fluid.initial_pressure * np.ones(self.field('pressure').shape)[:, :]
        self.field('friction')[:, :] = np.zeros(self.field('friction').shape)[:, :]
        self.field('speed_of_sound')[:, :] = np.zeros(self.field('friction').shape)[:, :]
        self.field('volume')[:, :] = self._volume0 * np.ones(self.field('volume').shape)[:, :]

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
        Exchange boundary values from previous time steps
        """
        # Exchange previous values with the left boundary
        self._pressure[1, 0] = self.left.value(Measure.pressureLast)
        self._velocity[1, 0] = self.left.value(Measure.velocityPlusLast)
        self._friction[1, 0] = self.left.value(Measure.frictionLast)
        self._sos[1, 1] = self.left.value(Measure.BPspeedOfSoundLast)
        self._area = self.left.value(Measure.area)

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

        area_a = self._area
        volume = self._volume[1, 1]
        pressure_a = self._pressure[1, 0]
        velocity_a = self._velocity[1, 0]
        friction_a = self._friction[1, 0]
        # Calculate fluid properties
        density_a = self.fluid.density(pressure=pressure_a, temperature=None)
        speed_of_sound = self._sos[1, 1]
        # Perform actual calculation

        f1 = (density_a * speed_of_sound * velocity_a
              + pressure_a
              - density_a * speed_of_sound * self._delta_t * friction_a)

        error = 1.0

        velocity = self._velocity[1, 1]

        while error > 1e-10:

            vol_fun = ((f1 - density_a * speed_of_sound * velocity)
                       * np.power((volume - velocity * area_a * self._delta_t), self._polytropic_exponent)
                       - self._pressure0 * np.power(self._volume0, self._polytropic_exponent)
                       )
            dvol_fun = (-density_a * speed_of_sound
                        * np.power((volume - area_a * self._delta_t * velocity), self._polytropic_exponent)
                        + (f1 - density_a * speed_of_sound * velocity)
                        * self._polytropic_exponent
                        * np.power((volume - velocity * area_a * self._delta_t), (self._polytropic_exponent - 1))
                        * (-area_a * self._delta_t)
                        )
            new_velocity = velocity - vol_fun / dvol_fun
            error = np.abs(new_velocity - velocity)
            velocity = new_velocity

        self._velocity[0, 1] = velocity
        self._pressure[0, 1] = f1 - density_a * speed_of_sound * velocity
        self._volume[0, 1] = (np.power((self._pressure0 / self._pressure[0, 1])
                                       * np.power(self._volume0, self._polytropic_exponent),
                                       1.0 / self._polytropic_exponent))

        return False
