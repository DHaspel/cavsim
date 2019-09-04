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
        super(Pipe, self).__init__(diameter, length, wall_thickness, bulk_modulus, roughness)
        # Register internal fields
        self._pressure: np.ndarray = self.field_create('pressure', 3)
        self._velocity: np.ndarray = self.field_create('velocity', 3)
        self.field_create('reynolds', 1)
        self.field_create('friction_factor', 1)
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
            ExportChannel(Measure.pressureLast2, lambda: self._pressure[2, 1]),
            ImportChannel(Measure.velocityPlusLast, False),
            ImportChannel(Measure.velocityPlusLast2, False),
            ExportChannel(Measure.velocityMinusCurrent, lambda: -self._velocity[0, 1]),
            ExportChannel(Measure.velocityMinusLast, lambda: -self._velocity[1, 1]),
            ExportChannel(Measure.velocityMinusLast2, lambda: -self._velocity[2, 1]),
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
            ExportChannel(Measure.pressureLast2, lambda: self._pressure[2, -2]),
            ImportChannel(Measure.velocityMinusLast, False),
            ImportChannel(Measure.velocityMinusLast2, False),
            ExportChannel(Measure.velocityPlusCurrent, lambda: self._velocity[0, -2]),
            ExportChannel(Measure.velocityPlusLast, lambda: self._velocity[1, -2]),
            ExportChannel(Measure.velocityPlusLast2, lambda: self._velocity[2, -2]),
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
        # todo: Calculate max delta t
        return None

    def discretize(self, delta_t: float) -> None:
        """
        Method handling the discretization of the component (for a given timestep width)

        :param delta_t: Timestep width to discretize for
        """
        self._delta_t = delta_t
        x_count = 5  # todo: discretize the internal states
        self.fields_resize(x_count)

    def initialize(self) -> None:
        """
        Initialize the internal state of the component (after discretization was called)
        """
        self.field('velocity')[:, :] = np.zeros(self._velocity.shape)[:, :]
        self.field('pressure')[:, :] = self.fluid.norm_pressure * np.ones(self._pressure.shape)[:, :]
        # todo: Initialize the internal states

    def prepare_next_timestep(self, delta_t: float, next_total_time: float) -> None:
        """
        Prepare the internal state for the next timestep to be calculated

        :param delta_t: Timestep width for the next timestep
        :param next_total_time: Total simulation time at the end of the next timestep
        """
        # Shift all internal fields
        self.fields_move()
        # Exchange previous values with the left boundary
        self._pressure[1, 0] = self.left.value(Measure.pressureLast)
        self._velocity[1, 0] = self.left.value(Measure.velocityPlusLast)
        self._velocity[2, 0] = self.left.value(Measure.velocityPlusLast2)
        # Exchange previous values with the right boundary
        self._pressure[1, -1] = self.right.value(Measure.pressureLast)
        self._velocity[1, -1] = -self.right.value(Measure.velocityMinusLast)
        self._velocity[2, -1] = -self.right.value(Measure.velocityMinusLast2)
        # todo: Other things to prepare each timestep

    def prepare_next_inner_iteration(self, iteration: int) -> None:
        """
        Method to prepare the internal state for the next inner iteration of the current timestep

        :param iteration: Number of the next inner iteration to prepare for
        """
        # todo: Exchange current properties with boundaries
        # todo: Other things to prepare each inner iteration

    def calculate_next_inner_iteration(self, iteration: int) -> bool:
        """
        Method to do the calculations of the next inner iteration

        :param iteration: Number of the next inner iteration
        :return: Whether this component needs another inner iteration afterwards
        """
        # todo: Perform the actual calculation
        return False
